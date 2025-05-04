import numpy as np
import pandas as pd
from PIL import Image
import time, os, pickle
from transformers import AutoModel
from .prompts import DefaultPrompt
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPModel, CLIPProcessor

class ClipEncoder:
    def __init__(self, model_path):
        self.MODEL_PATH = model_path
        self.mode_type = model_path.split("/")[-1]
        print(f"Loading model: {self.mode_type}")
        if self.mode_type == "clip-vit-large-patch14":
            self.model = CLIPModel.from_pretrained(model_path)
            self.processor = CLIPProcessor.from_pretrained(model_path)
        elif self.mode_type == "jina-clip-v1":
            self.model = AutoModel.from_pretrained(self.MODEL_PATH, trust_remote_code=True)
    
    def resize_image(self, image_path, size=(224, 224)):
        img_pil = Image.open(image_path).convert("RGB")
        return img_pil.resize(size)

    def text_encode(self, text):
        if self.mode_type == "clip-vit-large-patch14":
            input = self.processor(text=text, return_tensors="pt", truncation=True, max_length=77)
            encoded_input = self.model.get_text_features(**input)
            return encoded_input.detach().numpy().squeeze(0)
        
        elif self.mode_type == "jina-clip-v1":
            return self.model.encode_text(text)
    
    def image_encode(self, image_path):
        if self.mode_type == "clip-vit-large-patch14":
            input = self.processor(images=self.resize_image(image_path), return_tensors="pt")
            encoded_input = self.model.get_image_features(**input)
            return encoded_input.detach().numpy().squeeze(0)
        

        elif self.mode_type == "jina-clip-v1":
            return self.model.encode_image(self.resize_image(image_path))

class FeaturesEncoder(ClipEncoder):
    def __init__(self, images_path, encoding_images_path, encoding_labels_path, 
                 img_file_type, model:str="../../models/jina-clip-v1"):
        super().__init__(model)
        self.images_path = images_path
        self.img_file_type = img_file_type
        self.encoding_images_path = encoding_images_path
        self.encoding_labels_path = encoding_labels_path

    def encode_images(self, img_features_path):
        img_features_df = pd.read_parquet(img_features_path)
        try:
            with open(self.encoding_images_path, "rb") as f:
                embedding_dict = pickle.load(f)
        except:
            print("Creating embedding dict...")
            embedding_dict = {}
            
        for i in range(len(img_features_df)):
            start = time.time()
            print(f'{i+1}/{len(img_features_df)}')

            if img_features_df['file_name'][i] in embedding_dict.keys():
                print(f"{img_features_df['file_name'][i]} done!")
            else:
                img_id = img_features_df['file_name'][i].split("_")[-1]
                folder = img_features_df['file_name'][i].split("_")[:-1]
                folder = "_".join(folder)
                new_path = os.path.join(self.images_path, folder, f"image_{img_id}.{self.img_file_type}")
                # print(f"New path: {new_path}")
                emb = {}
                emb['file_name'] = img_features_df['file_name'][i]
                emb['label_id'] = img_features_df['label_id'][i]
                emb['init_pred'] = self.text_encode(img_features_df['init_pred'][i])
                emb['img_desc'] = self.text_encode(img_features_df['img_desc'][i])
                # emb['img_emb'] = self.image_encode(os.path.join(self.images_path, f"{img_features_df['file_name'][i]}.{self.img_file_type}"))
                emb['img_emb'] = self.image_encode(new_path)

                # Save to dict
                embedding_dict[img_features_df['file_name'][i]] = emb
                # Save to file
                with open(self.encoding_images_path, "wb") as f:
                    pickle.dump(embedding_dict, f)

                end = time.time()
                print(f"Time taken per label: {round(end - start, 2)} seconds")
                print("-"*50)

    def encode_labels(self, labels_features_path, human_design_prompt):
        class_info_df = pd.read_parquet(labels_features_path)

        try:
            with open(self.encoding_labels_path, "rb") as f:
                embedding_dict = pickle.load(f)
        except:
            print("Creating embedding dict...")
            embedding_dict = {}

        for i in range(len(class_info_df)):
            start = time.time()
            print(f'{i+1}/{len(class_info_df)}')

            if class_info_df['label'][i] in embedding_dict.keys():
                print(f"{class_info_df['label'][i]} done!")
            else:
                emb = {}
                # Class Description - Embeddings
                avg = []
                columns = [
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
                    ]
                for col in columns:
                    encoded = self.text_encode(class_info_df[col][i])
                    avg.append(encoded)
                    emb[col] = encoded
                # Class Description - Fused Embedding
                emb['CDFE'] = np.mean(np.stack(avg), axis=0)

                # Class Embedding
                emb['CE'] = self.text_encode(class_info_df['label'][i])
                # Human Design Embedding
                emb['HDE'] = self.text_encode(human_design_prompt.format(class_info_df['label'][i]))
                # Description Features Embedding
                emb['DF'] = np.mean(np.stack([emb['CDFE'], emb['CE'], emb['HDE']]), axis=0)
                # Save to dict
                embedding_dict[class_info_df['label'][i]] = emb
                # Save to file
                with open(self.encoding_labels_path, "wb") as f:
                    pickle.dump(embedding_dict, f)
                end = time.time()
                print(f"Time taken per label: {round(end - start, 2)} seconds")
                print("-"*50)