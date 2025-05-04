import base64

class LLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise ValueError(f"Image file at {image_path} not found.")
        except Exception as e:
            raise ValueError(f"An error occurred while encoding the image: {e}")

    def multimodal_generate(self, image_path, prompt=None, temp=0.1):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def text_generate(self, prompt=None, temp=0.99):
        raise NotImplementedError("This method should be implemented in a subclass.")
    
# --------------------------------------------------------------
class OpenAIModel(LLM):
    def __init__(self, api_key, model_name='gpt-4.1-nano-2025-04-14'):
        from openai import OpenAI
        # model of: gpt-4o-mini, gpt-4.1-nano-2025-04-14
        super().__init__(model_name)
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)

    def multimodal_generate(self, image_path, prompt=None, temp=0.1):
        base64_image = self.encode_image(image_path)
        completion = self.client.chat.completions.create(
            model=self.model_name, 
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature = temp,
        )
        return completion.choices[0].message.content

    def text_generate(self, prompt=None, temp=0.99):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=temp,
        )
        return completion.choices[0].message.content
    
# --------------------------------------------------------------
class LlavaModel(LLM):
    def __init__(self, api_token, model_name='yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb'):
        from replicate.client import Client
        super().__init__(model_name)
        self.api_token = api_token
        self.client = Client(api_token=self.api_token)

    def multimodal_generate(self, image_path, prompt=None, temp=0.1):
        base64_image = self.encode_image(image_path)
        output = self.client.run(
            self.model_name,
            input = {
                "image": f"data:image/jpeg;base64,{base64_image}",
                "prompt": prompt,
                "temperature": temp,
            }
        )
        response = ("".join(output))
        return response
    
    def text_generate(self, prompt=None, temp=0.99):
        raise NotImplementedError("LlavaModel does not support text generation. Use multimodal_generate with an image input.")

# --------------------------------------------------------------
class LlamaModel(LLM):
    def __init__(self, api_token, model_name='meta/meta-llama-3-70b-instruct'):
        from replicate.client import Client
        super().__init__(model_name)
        self.api_token = api_token
        self.client = Client(api_token=self.api_token)

    def multimodal_generate(self, image_path, prompt=None, temp=0.1):
        raise NotImplementedError("LlamaModel does not support image as input.")

    def text_generate(self, prompt=None, temp=0.99):
        input = {
            "top_p": 0.9,
            "prompt": prompt,
            "min_tokens": 0,
            "temperature": temp,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15
        }

        output = self.client.run(
            self.model_name,
            input=input
        )
        return ("".join(output))

# --------------------------------------------------------------
class GeminiModel(LLM):
    def __init__(self, api_key, model_name='gemini-2.0-flash-exp'):
        from google import genai
        # model of: gemini-2.0-flash-exp, gemini-2.0-flash-lite-001, 
        # gemini-2.0-flash-001, gemini-1.5-flash-8b-001, gemini-1.5-flash-002
        super().__init__(model_name)
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    def multimodal_generate(self, image_path, prompt=None, temp=0.1):
        from PIL import Image
        from google.genai import types
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=temp
            )
        )
        return response.text

    def text_generate(self, prompt=None, temp=0.99):
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=temp
            )
        )
        return response.text

# --------------------------------------------------------------
"""
    import os
    from modules.llm import OpenAIModel, GeminiModel, LlavaModel
    from dotenv import load_dotenv; load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OpenAI = OpenAIModel(OPENAI_API_KEY, model_name='gpt-4.1-nano-2025-04-14')

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    Gemini = GeminiModel(GEMINI_API_KEY, model_name='gemini-1.5-flash-8b-001')

    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    LLaVA = LlavaModel(REPLICATE_API_TOKEN)

    # --------------------------------------------------------------
    prompt = "Describe the image"
    image_path = '/Users/dna-tuananguyen/Downloads/Project5922/database/Pets/images/Abyssinian_1.jpg'
    example_1 = Gemini.multimodal_generate(image_path, prompt)
    example_2 = LLaVA.multimodal_generate(image_path, prompt)
    example_3 = OpenAI.multimodal_generate(image_path, prompt)

    prompt_2 = "What is the capital of France?"
    example_4 = Gemini.text_generate(prompt_2)
    example_5 = OpenAI.text_generate(prompt_2)
"""
