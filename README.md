# AudioML - Zero To Hero
A deep dive into the current state of audio machine learning - from zero to hero.
## About
Hi! I am an AI Engineer working a lot with audio-based models and want to go more deeply into the field, instead of just scratching the surface.

## Timeline:
### 29.03.2026
Today I worked through the blog https://kyutai.org/codec-explainer and let Sonnet 4.6 make me some homework programming tasks. 

I had prior understanding of codebooks, but implementing the different RVQ levels from scratch and plotting them made it really click how longer codebooks work. RVQ gives me the vibe of different levels of differentiation going from distance → velocity → acceleration (dx/dt), like always going one level deeper of information.

Started implementing a model based on EnCodec by reading the paper https://arxiv.org/pdf/2210.13438. 

### 30.03.2026
Implemented Decoder, RVQ, and loss function -> next task: training loop

### 31.03.2026 - 02.04.2026
Implemented some of the loss functions, not all, and trained on the Gemini speech dataset. Very bad results — loss wasn't going down. Reconstruction doesn't work at all.

### 03.04.2026

I played around with the losses, used log of the mels instead of mel itself to lower the values of the loss.

Debugging why the model wasn't learning anything: the model just kept using a very low amount of the codebook, therefore:

I added something similar to EMA to replace dead codebooks and added entropy loss to drive the embeddings apart. Finally it's learning something haha :D and cleaned up the file overall.

ITS LEARNING YES

