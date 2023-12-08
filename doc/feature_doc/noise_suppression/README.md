# OpenVINO™ Noise Suppression :broom:

This feature removes background noise from an audio sample containing spoken audio.  

It can be found under the **Effect** menu:
![](menu.png)  

## Description of properties 
See below for a description of the properties that can be set for this effect:  
![](properties.png)  
1. **OpenVINO Inference Device**:  The OpenVINO™ device that will be used to run the noise suppression model. 
2. **Noise Suppression Model**: Used to select which noise-suppression model to use. Right now we only support *denseunet*, but support of others may be added in the future.

After clicking *Apply*, you'll see this dialog window pop up:  
![](loading.png)

At this stage, the noise suppression AI model is getting loaded to the chosen device (e.g. CPU, GPU, NPU, etc.). This usually takes 10 to 30 seconds if it's the first time running with this device after installing these plugins since it needs to compile the model specifically for the device you've chosen. These *compiled* model will be cached on disk though -- so it should run much faster the next time that it is loaded.

This effect directly modifies the selected track(s).
