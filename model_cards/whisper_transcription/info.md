## Quantization Guide

The Whisper models available to use with this project come in three *quantization* variants: *FP16*, *INT8*, & *INT4*.

**FP16**: Floating-Point 16 (i.e. half precision, 16-bit). Highest memory footprint & utilization, but gives highest accuracy (transcription quality) in most cases. Lowest performance (on most systems).  

**INT8**: Integer-8 (8-bit) quantized. Roughly 1/2 of the memory footprint / utilization as compared with FP16. Improved performance on most systems, as compared with *FP16*. Minimal loss in accuracy (transcription quality) as compared with *FP16*.  

**INT4**: Integer-4 (4-bit) quantized. Roughly 1/4 of the memory footprint / utilization as compared with FP16. Improved performance on most systems, as compared with *INT8* & *FP16*. Minimal loss in accuracy (transcription quality) as compared with *INT8* & *FP16*.  

## Model Variant Guide 

**Note**: The following guide has used / copied / modified some of the information found here: [https://whisper-api.com/blog/models/](https://whisper-api.com/blog/models/)

There are many variants of whisper models, each one representing a tradeoff between accuracy, speed, and resource requirements.

### Base:
A 74M parameter multilingual model. 
**Multi-Language Support**: Yes  
**Best for**: General purpose transcription with reasonable accuracy when resources are limited. This is a great balance between speed and accuracy.  
 
### Small:  
A 244M parameter multilingual model.  
**Multi-Language Support**: Yes  
**Best for**: Daily transcription needs with good accuracy and reasonable speed. More accurate than base model but requires more resources.  
  
### Medium:  
A 769M parameter multilingual model.  
**Multi-Language Support**: Yes  
**Best for**: High-quality transcriptions where accuracy is important and you have decent computing resources.  

### Large:
There are (currently) three variants of Whisper-large:  
**large-v1**: Original large model (1.5B parameters)  
**large-v2**: Improved large model  
**large-v3**: Latest larger model with the best accuracy  (recommended). 

**Multi-Language Support**: Yes  
**Best for**: Professional transcription where maximum accuracy is essential, especially for *multi-language* transcriptions. 

### Distil-Large-V3:  
A *distilled* (756M parameter) version of the **large-v3** model, which can produce *large-v3*-like quality but at a dramatically reduced memory & processing cost. **But**.. it can only transcribe *english* recordings.  

**Multi-Language Support**: No (English-only transcription)  
**Best for**: Professional *english* transcription where maximum accuracy is essential.  
 
## How to use
There is some documentation [here](https://github.com/intel/openvino-plugins-ai-audacity/tree/main/doc/feature_doc/whisper_transcription) about how to use these plugins.

## OpenVINO Speech-to-Text collection

For use with [OpenVINO AI Plugins for Audacity](https://github.com/intel/openvino-plugins-ai-audacity), all of these models are downloaded from the OpenVINO's [Speech-to-Text](https://huggingface.co/collections/OpenVINO/speech-to-text-672321d5c070537a178a8aeb) collection on HuggingFace.

## License:

For *Distil* variants: [MIT](https://github.com/huggingface/distil-whisper/blob/main/LICENSE)  

For *Base*, *Small*, *Medium*, and *Large* variants: [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)  