import numpy as np
import pandas as pd

class ClassifierModel:
    def __init__(self, embedding_dict, mode:str='M4'):
        self.mode = mode
        self.class_names = list(embedding_dict.keys())
        self.M1weights = np.vstack([embedding_dict[label]['CE'] for label in self.class_names]).T
        self.M2weights = np.vstack([embedding_dict[label]['HDE'] for label in self.class_names]).T
        self.M3weights = np.vstack([embedding_dict[label]['CDFE'] for label in self.class_names]).T
        self.M4weights = np.vstack([embedding_dict[label]['DF'] for label in self.class_names]).T
        self.weights = self.get_model_weights()

    def get_model_weights(self):
        if self.mode == 'M1':
            print(f"Using model M1: Raw Class Embedding")
            return self.M1weights
        elif self.mode == 'M2':
            print(f"Using model M2: Human Design Embedding")
            return self.M2weights
        elif self.mode == 'M3':
            print(f"Using model M3: Class Description Features Embedding")
            return self.M3weights
        elif self.mode == 'M4':
            print(f"Using model M4: Fused Features Embedding")
            return self.M4weights
        else:
            raise ValueError("Invalid model choice. Choose between 1, 2, 3, or 4.")
    
# img_features -> ImageFileName -> {filename, label_id, init_pred, img_desc, img_emb}
class ImageClassifier(ClassifierModel):
    def __init__(self, embedding_dict, img_features, mode:str='M4', ifeature:str='X_if'):
        super().__init__(embedding_dict, mode)
        self.ifeature = ifeature
        self.img_features = img_features
        self.num_images = len(img_features)
        self.image_ids = list(img_features.keys())
        noti_ifeature = {'X_if' : 'Using Image Feature: Encoded Image',
                        'X_df' : 'Using Image Feature: Encoded Image Description',
                        'X_pf' : 'Using Image Feature: Encoded Init Prediction',
                        'X_q' : 'Using Image Feature: Encoded Fused Image Feature'
                        }
        try:
            print(noti_ifeature[self.ifeature], self.ifeature)
        except KeyError:
            raise ValueError("Invalid ifeature type. Choose between 'X_if', 'X_df', 'X_pf', or 'X_q'.")

    def get_data(self, image):
        image_feature = None
        if self.ifeature == 'X_if':
            image_feature = self.img_features[image]['img_emb']
            image_feature /= np.linalg.norm(image_feature)
        elif self.ifeature == 'X_df':
            image_feature = self.img_features[image]['img_desc']
            image_feature /= np.linalg.norm(image_feature)
        elif self.ifeature == 'X_pf':
            image_feature = self.img_features[image]['init_pred']
            image_feature /= np.linalg.norm(image_feature)
        elif self.ifeature == 'X_q':
            image_feature = self.img_features[image]['img_emb'] + self.img_features[image]['img_desc'] + self.img_features[image]['init_pred']
            image_feature /= np.linalg.norm(image_feature)
        else:
            raise ValueError("Invalid feature type. Choose between 'X_if', 'X_df', 'X_pf', or 'X_q'.")
        
        return image_feature

    def predicting_df(self):
        df = pd.DataFrame(columns=['image', 'true_label', 'final_pred'])
        for image in self.image_ids:
            df = pd.concat([df, pd.DataFrame({
                'image': [image],
                'true_label': [self.img_features[image]['label_id']],
                'final_pred': [None]
            })], ignore_index=True)
        return df
    
    def classify(self):
        df = self.predicting_df()
        for i in range(len(df)):
            image_feature = self.get_data(df['image'][i])
            index = np.argmax(np.matmul(image_feature, self.weights))
            final_pred = self.class_names[index.squeeze()]
            df.at[i, 'final_pred'] = final_pred
        return df
    
    def metrics(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return accuracy, precision, recall, f1
    
    def evaluation(self, df):
        accuracy, recall, precision, f1 = self.metrics(df['true_label'], df['final_pred'])
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        return accuracy, precision, recall, f1