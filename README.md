# Kohya Trainer V4 Colab UI - VRAM 12GB
### Best way to train Stable Diffusion model for peeps who didn't have good GPU

Adapted to Google Colab based on [Kohya Guide](https://note.com/kohya_ss/n/nbf7ce8d80f29#c9d7ee61-5779-4436-b4e6-9053741c46bb) <br>
Adapted to Google Colab by [Linaqruf](https://github.com/Linaqruf)<br>
You can find the latest notebook update [here](https://github.com/Linaqruf/kohya-trainer/blob/main/kohya-trainer.ipynb)

---
## What is this?
---
### **_Q: So what's the difference between `Kohya Trainer` and other Stable Diffusion trainers out there?_**
### A: **Kohya Trainer** has some new features like
1. Using the U-Net learning
2. Automatic captioning/tagging for every image automatically with BLIP/DeepDanbooru
3. Implementing the [NovelAI Aspect Ratio Bucketing Tool](https://github.com/NovelAI/novelai-aspect-ratio-bucketing) so you don't need to crop image datasets to 512x512 anymore
- Using the output of the second-to-last layer of CLIP (Text Encoder) instead of the last layer.
- Learning at non-square resolutions (Aspect Ratio Bucketing).
- Extending token length from 75 to 225.
4. By preparing a certain number of images (several hundred or more seems to be desirable), you can make learning even more flexible than with DreamBooth.
5. It also supports Hypernetwork learning
6. `NEW!` 23/11 - Implemented Waifu Diffusion 1.4 Tagger for alternative DeepDanbooru for auto-tagging
7. 

### **_Q: And what's the difference between this notebook and other DreamBooth notebooks out there?_**
### A: We're adding Quality of Life features such as:
- Installing **gallery-dl** to scrape images, so you can get your own dataset quickly with Google bandwidth
- Huggingface Integration, where you can log in to Huggingface-hub and upload your trained model/dataset to Huggingface
---

## Credit
[Kohya](https://twitter.com/kohya_ss) | Just for my part
