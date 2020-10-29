# recognition_handler.py
from _recognition_handler import RecognitionHandler

_service = RecognitionHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
        
    if data is None:
        return None
    
    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)
    
    return data