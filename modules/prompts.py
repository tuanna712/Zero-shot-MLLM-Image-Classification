

class Prompt:
    def __init__(self, name):
        self.name = name
    
    def human_design(self):
        return "A photo of {}"
    
    def class_desc(self):
        return [
            "Describe what a {} looks like in one or two sentences.",
            "How can you identify a {} in one or two sentences?",
            "What does a {} look like? Respond with one or two sentences.",
            "Describe an image from the internet of a {}. Respond with one or two sentences.",
            "A short caption of an image of a {}:"
        ]
    
    def image_desc(self):
        return "What do you see? Describe any object precisely, including its type or class."
    
    def image_clf(self):
        return "You are given an image and a list of class labels. Classify the image given the class labels. " \
        "Answer using a single word if possible. Only choose word in class list. Here are the class labels: {}"
    
class DefaultPrompt(Prompt):
    def __init__(self, name):
        super().__init__(name)
        self.human_design = self.human_design()
        self.class_desc = self.class_desc()
        self.image_desc = self.image_desc()
        self.image_clf = self.image_clf()
    
    def all_prompts(self):
        return {
            "human_design": self.human_design,
            "class_desc": self.class_desc,
            "image_desc": self.image_desc,
            "image_clf": self.image_clf
        }
    
class CoTPrompt(Prompt):
    def __init__(self, name):
        super().__init__(name)
        self.human_design = "A photo of {}"
        self.class_desc = [
            "Describe what a {} looks like in one or two sentences.",
            "How can you identify a {} in one or two sentences?",
            "What does a {} look like? Respond with one or two sentences.",
            "Describe an image from the internet of a {}. Respond with one or two sentences.",
            "A short caption of an image of a {}:"
        ]
        self.image_desc = "First, identify the main object in the image. Then, describe its visual attributes " \
                "such as shape, color, and size. Finally, infer the most likely category or class based on these observations. "\
                "Think step by step and provide a concise description of the object in 2-3 sentences. "\
                "Directly describe without any additional expression. Dont need to mention: Here's a breakdown."\
                "Directly Answer: Main Object...Visual Attributes...Category or Class"
        
        self.image_clf = "You are given an image and a list of class labels. "\
                "Think step by step. First, describe key visual features of the image. "\
                "Then, reason which class label best fits the visual characteristics. "\
                "Finally, dont return the thinking steps, just answer using a single word if possible. "\
                "Here are the class labels:{}"
    
    def all_prompts(self):
        return {
            "human_design": self.human_design,
            "class_desc": self.class_desc,
            "image_desc": self.image_desc,
            "image_clf": self.image_clf
        }
    
class ReasonCoTPrompt(Prompt):
    def __init__(self, name):
        super().__init__(name)
        self.human_design = self.human_design()
        self.class_desc = self.class_desc()
        self.image_desc = self.image_desc()
        self.image_clf = self.image_clf()
    
    def all_prompts(self):
        return {
            "human_design": self.human_design,
            "class_desc": self.class_desc,
            "image_desc": self.image_desc,
            "image_clf": self.image_clf
        }
    
class GoTPrompt(Prompt):
    def __init__(self, name):
        super().__init__(name)
        self.human_design = self.human_design()
        self.class_desc = self.class_desc()
        self.image_desc = self.image_desc()
        self.image_clf = self.image_clf()
    
    def all_prompts(self):
        return {
            "human_design": self.human_design,
            "class_desc": self.class_desc,
            "image_desc": self.image_desc,
            "image_clf": self.image_clf
        }