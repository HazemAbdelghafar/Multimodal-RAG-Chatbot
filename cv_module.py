from transformers import BlipProcessor, BlipForQuestionAnswering

def initialize_visual_answering():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, blip_model


def invoke_visual(processor, blip_model, image, question):    
    # Process the image and question
    inputs = processor(image, question, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    response = processor.decode(outputs[0])[:-6] 
    return response
