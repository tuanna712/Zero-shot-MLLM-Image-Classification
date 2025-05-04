import time, os, json
import pandas as pd
from .prompts import DefaultPrompt
DefPrompt = DefaultPrompt("Defaut")

class Dataset:
    def __init__(self, images_path, class_meta_path):
        self.images_path = images_path
        self.class_meta_path = class_meta_path
        self.meta = json.load(open(class_meta_path, "r"))
        self.classes = list(self.meta['labels'].keys())
    
    def json_to_df(self, json_data):
        # Convert the classes dictionary to a DataFrame
        df = pd.DataFrame.from_dict(json_data, orient='columns').reset_index()
        # df = df.rename(columns={"index": "name"})
        return df#.sort_values(by="name").reset_index(drop=True)
    
class Pets(Dataset):
    def __init__(self, images_path, class_meta_path):
        super().__init__(images_path, class_meta_path)

class Caltech101(Dataset):
    def __init__(self, images_path, class_meta_path):
        super().__init__(images_path, class_meta_path)
    
class ImageFeatures(Dataset):
    def __init__(self, images_path, class_meta_path, working_images, llm_model, PromptSys=DefPrompt):
        super().__init__(images_path, class_meta_path)
        self.llm = llm_model
        self.Prompt = PromptSys
        self.images_path = images_path
        self.working_images = working_images
        self.load_working_data(self.working_images)

    def load_working_data(self, working_images):
        try:
            self.img_features = pd.read_parquet(working_images)
            print("Loaded parquet!")
        except:
            print("Dont find saved file! Creating new data...")
            self.img_features = pd.DataFrame(columns=['file_name', 'label_id', 'init_pred', 'img_desc'])

            self.img_features['file_name'] = list(self.meta['images'].keys())
            self.img_features['label_id'] = [v['label'] for k, v in self.meta['images'].items()]
            print("Created dataframe!")
            self.img_features.to_parquet(working_images)
            print("Saved parquet!")

    def gen_info(self):
        self.load_working_data(self.working_images)
        for i in range(len(self.img_features)):
            start = time.time()
            print(f'{i+1}/{len(self.img_features)}')
            if self.img_features['img_desc'][i] is not None or pd.notna(self.img_features.loc[i, 'img_desc']):
                print(f"{self.img_features['file_name'][i]} done!")
            else:
                current_img = self.img_features['file_name'][i]
                print(f"Image: {self.img_features['file_name'][i]}")
                image_path = os.path.join(self.images_path, f"{self.meta['images'][current_img]['file_path']}")
                time.sleep(15)
                self.img_features.loc[i, 'init_pred'] = self.llm.multimodal_generate(
                                                                image_path,
                                                                self.Prompt.image_clf.format(self.classes))
                self.img_features.loc[i, 'img_desc'] = self.llm.multimodal_generate(
                                                                image_path,
                                                                self.Prompt.image_desc)
            
                # Save file
                self.img_features.to_parquet(self.working_images)
                end = time.time()
                print(f"Time taken per label: {round(end - start, 2)} seconds")
                print("-"*50)

class LabelFeatures(Dataset):
    def __init__(self, images_path, class_meta_path, working_labels, llm_model, PromptSys=DefPrompt):
        super().__init__(images_path, class_meta_path)
        self.llm = llm_model
        self.Prompt = PromptSys
        self.working_labels = working_labels
        self.load_working_data()

    def load_working_data(self):
        try:
            self.label_features = pd.read_parquet(self.working_labels )
            print("Loaded parquet!")
        except:
            print("Creating dataframe...")
            self.label_features = pd.DataFrame(columns=['label', 
                'CDR_10', 'CDR_20', 'CDR_30', 'CDR_40', 'CDR_50',
                'CDR_11', 'CDR_21', 'CDR_31', 'CDR_41', 'CDR_51',
                # 'CDR_12', 'CDR_22', 'CDR_32', 'CDR_42', 'CDR_52',
                # 'CDR_13', 'CDR_23', 'CDR_33', 'CDR_43', 'CDR_53',
                # 'CDR_14', 'CDR_24', 'CDR_34', 'CDR_44', 'CDR_54',
                # 'CDR_15', 'CDR_25', 'CDR_35', 'CDR_45', 'CDR_55',
                # 'CDR_16', 'CDR_26', 'CDR_36', 'CDR_46', 'CDR_56',
                # 'CDR_17', 'CDR_27', 'CDR_37', 'CDR_47', 'CDR_57',
                # 'CDR_18', 'CDR_28', 'CDR_38', 'CDR_48', 'CDR_58',
                # 'CDR_19', 'CDR_29', 'CDR_39', 'CDR_49', 'CDR_59'
                ])
            
            self.label_features['label'] = self.classes
            print("Created dataframe!")
            self.label_features.to_parquet(self.working_labels)
            print("Saved parquet!")

    def generate_class_descriptions(self, label, prompts):
        descriptions = {}
        for idx, prompt in enumerate(prompts, start=1):
            descriptions[f'CDR_{idx}0'] = self.llm.text_generate(prompt.format(label))
            descriptions[f'CDR_{idx}1'] = self.llm.text_generate(prompt.format(label))
            # descriptions[f'CDR_{idx}2'] = self.llm.text_generate(prompt.format(label))
            # descriptions[f'CDR_{idx}3'] = self.llm.text_generate(prompt.format(label))
            # descriptions[f'CDR_{idx}4'] = self.llm.text_generate(prompt.format(label))
            # descriptions[f'CDR_{idx}5'] = self.llm.text_generate(prompt.format(label))
            # descriptions[f'CDR_{idx}6'] = self.llm.text_generate(prompt.format(label))
            # descriptions[f'CDR_{idx}7'] = self.llm.text_generate(prompt.format(label))
            # descriptions[f'CDR_{idx}8'] = self.llm.text_generate(prompt.format(label))
            # descriptions[f'CDR_{idx}9'] = self.llm.text_generate(prompt.format(label))
        return descriptions
    
    def gen_info(self):
        self.load_working_data()
        # Check worked labels
        for i in range(len(self.label_features)):
            start = time.time()
            print(f'{i+1}/{len(self.label_features)}')
            if self.label_features['CDR_51'][i] is not None or pd.notna(self.label_features.loc[i, 'CDR_51']):
                print(f"Label {i} done!")
            else:
                time.sleep(10)
                # Generate class descriptions
                descriptions = self.generate_class_descriptions(
                    self.label_features.loc[i, 'label'], 
                    self.Prompt.class_desc
                )
                for key, value in descriptions.items():
                    self.label_features.loc[i, key] = value

                # Save working file
                self.label_features.to_parquet(self.working_labels)
                print("Saved working file!")
            end = time.time()
            print(f"Time taken per label: {round(end - start, 2)} seconds")
            print("-"*50)
