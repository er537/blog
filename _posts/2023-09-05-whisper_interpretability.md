---
layout: post
title: Interpretting Whisper
---
*(Work done as part of [SERI MATS](https://www.serimats.org/) Summer 2023 cohort under the supervision of Lee Sharkey.)*


Thus far, Mechanistic Interpretability has primarily focused on language and image models. To develop a universal model for interpretability we need techniques that transfer across modalities. I have therefore attempted to do mechanistic interpretability on OpenAI's Whisper **tiny** model and here I present some of my findings. This post is structured into 3 main claims that I make about the model:


**1) Encoder representations are highly localized  
2) The encoder learns human interpretable features  
3) The decoder alone acts as a weak LM**  

*For context: Whisper is a speech-to-text model. It has an encoder-decoder transformer architecture and Whisper tiny is only 39M parameters! The input to the encoder is a 30s chunk of audio (shorter chunks can be padded) and the ouput from the decoder is the transcript, predicted autoregressively. It is trained only on labelled speech to text pairs.*

# 1) Encoder representations are highly localized  

Below I present 3 experiments that suggest that the representations at the output of the decoder are highly localized; that is, they do not use much information from sequence positions outside of a narrow attention window. This is in contrast to a standard LLM which often attends to source tokens far away from the destination token.

## Attention patterns are very localized
We propagate the attention scores $R_{t}$ down the layers of the encoder as in [Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers](https://arxiv.org/pdf/2103.15679.pdf). This roughly equates to,

<div align="center">
  $$R_{t+1} = R_{t} + \bar A_{t+1} R_{t},$$
</div>


where,  
<div align="center">
  $$\bar A_t = \mathbb{E}[\nabla A_t \circ A_t],$$  
</div>


$A_{t}$ is the attention pattern in layer $t$ and $\bar A_{t}$ is the attention pattern weighted by gradient contribution. 
This produces the striking pattern below; up to the point where the audio ends, the attention pattern is very localized. When the speech ends (at frame ~500 in the following plot), all future positions attend back to the end of the speech.
<div style="display: flex; justify-content: center;">
    <img src="/blog/assets/images/whisper_interpretability/encoder/attention_scores.png" alt="attn_scores" style="max-width: 100%; height: auto;" />
</div>

## Constraining the attention window
Given how localized the attention pattern appears to be, we investigate what happens if we constrain it so that every audio embedding can only attend to the k nearest tokens on either side. Eg if k=2 we would we apply the following mask to the attention scores before the softmax:
<div>
    <img src="/blog/assets/images/whisper_interpretability/encoder/attn_mask.png" alt="attn_mask" style="width:300px; height:300px;" />
</div>

Here are the transcripts that emerge as we limit the attention window for various k values. We observe that even when k is reduced to 75, the model continues to generate reasonably precise transcripts, indicating that information is being encoded in a localized manner.



##### Original transcript (seq_len=1500):  
'hot ones. The show where celebrities answer hot questions while feeding even hotter wings.' 


##### k=100:  
"Hot ones. The show where celebrities answer hot questions, what feeding, eating hot wings. I am Shana Evans. I'm Join Today." 


##### k=75:  
"The show with celebrities and their hot questions, what feeding, eating hot wings. Hi, I'm Shannon, and I'm joined today." 


##### k=50:  
'The show where celebrities enter hot questions, what leading, what leading, what are we.' 


##### k=20:  
"I'm joined today."


##### k=10:  
""

## Removing words in embedding space
Recall that Whisper is an encoder-decoder transformer; the decoder cross-attends to the output of the final layer of the encoder. Given the apparent localization of the embeddings in this final layer, we postulate that we could remove words from the transcript by 'chopping' them out in embedding space. Concretely we let,

`final_layer_output[start_index:stop_index] = final_layer_output_for_padded_input[start_index:stop_index]`,


 where `final_layer_output_for_padded_input` is the output of the encoder when we just use padding frames as the input.

Consider the following example in which we substitute the initial 50 audio embeddings with padded equivalents (e.g., start_index=0, stop_index=50). These 50 embeddings represent $(50/1500)*30s=1s$ of audio. Our observation reveals that the transcript resulting from this replacement omits the initial two words.

<details>
<summary>Audio Sample</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/encoder/Hot_ones.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>


##### Original Transcript:
`hot ones. The show where celebrities answer hot questions while feeding even hotter wings.`  
##### Substitute embedding between (start_index=0, stop_index=50):     
`The show where celebrities answer hot questions while feeding even hotter wings.`   


We can also do this in the middle of the sequence. Here we let (start_index=150, stop_index=175) which corresponds to 3-3.5s in the audio and observe that the transcipt omits the words `hot questions`:  

##### Original:   
`hot ones. The show where celebrities answer hot questions while feeding even hotter wings.`  
##### Substitute embeddings between (start_index=150, stop_index=175):  
`hot ones. The show where celebrities while feeding even hotter wings.` 


# 2) The encoder learns human interpretable features
## Neuron Basis
It turns out that neurons in the MLP layers of the encoder are highly interpretable; by finding maximally activating dataset examples (from a dataset of 10,000 2s audio clips) for all of the neurons we found that the majority activate on a specific phonetic sound! The table below shows these sounds for the first 50 neurons in `block.2.mlp.1`. By amplifying the audio around the sequence position where the neuron is maximally active, you can clearly hear these phonemes, as demonstrated by the audio clips below. 

| Neuron idx | 0   | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   |
| ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phoneme    | 'm' | 'j/ch/sh'  | 'e/a'  | 'c/q'  | 'is'  | 'i'  | noise  | 'w'  | 'l'  | 'the'  |

| Neuron idx | 10  | 11  | 12  | 13  | 14  | 15  | 16  | 17  | 18  | 19  |
| ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phoneme    | 'I' | N/A | noise  | vowels  | r  | st  | I | N/A  | ch  | p |

| Neuron idx | 30  | 31  | 32  | 33  | 34  | 35  | 36  | 37  | 38  | 39  |
| ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phoneme    | N/A | N/A  | 'd'  | 'p'  | 'n'  | 'q'  | 'a'  | 'A/E/I'  | *microphone turning on*  | 'i'  |

|Neuron idx | 40  | 41  | 42  | 43  | 44  | 45  | 46  | 47  | 48  | 49  |
| ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phoneme  | 's' | N/A  | 'air'  | 'or/all'  | 'e/i'  | 'th'  | N/A  | 'w' | 'eer' | 'w' |

### encoder.blocks.3.mlp.1 - Neuron Basis
<details>
<summary>Neuron 4 ('said')</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_4pkl_sample0.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_4pkl_sample3.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_4pkl_sample4.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_4pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Neuron 9 ('way/wait')</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_9pkl_sample5.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_9pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_9pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Neuron 20 ('f')</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_20pkl_sample0.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_20pkl_sample6.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_20pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_mlp1_nb_20pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

### encoder.blocks.2.mlp.1 - Neuron Basis
<details>
<summary>Neuron 0 ('m')</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_0pkl_sample3.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_0pkl_sample6.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_0pkl_sample7.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_0pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Neuron 1 ('sh/ch'))</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_1pkl_sample1.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_1pkl_sample2.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_1pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_1pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Neuron 3 ('c/q')</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_3pkl_sample4.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_3pkl_sample5.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_3pkl_sample6.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp1_nb_3pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

## Residual Stream Features
The residual stream is not in a [privileged basis](https://transformer-circuits.pub/2023/privileged-basis/index.html) so we would not expect the features it learns to be neuron aligned. Instead we trained [sparse autoencoders](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition) on the residual stream activations and found maximally activating dataset examples for these learnt features. Below are some examples of these features:


### encoder.blocks.3 - Learnt using sparse autoencoder
<details>
<summary>Dictionary idx=131 ("r")</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_131pkl_sample2.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_131pkl_sample7.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_131pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_131pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Dictionary idx=1 ("n")</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_1pkl_sample0.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_1pkl_sample1.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_1pkl_sample3.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_1pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Dictionary idx=2 ("p")</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_2pkl_sample0.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_2pkl_sample2.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_2pkl_sample3.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b3_res_2pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

### encoder.blocks.2 - Learnt using sparse autoencoder
<details>
<summary>Dictionary idx=3 ("an/in/on")</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_3pkl_sample1.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_3pkl_sample6.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_3pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_3pkl_sample8.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Dictionary idx=4 ("I (eg time/try/I))</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_4pkl_sample3.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_4pkl_sample6.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_4pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_4pkl_sample7.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Dictionary idx=6 ("l" (eg lost))</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_6pkl_sample0.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_6pkl_sample2.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_6pkl_sample9.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_res_6pkl_sample3.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

# Polysemantic acoustic neurons
The presence of polysemantic neurons in both language and image models is widely acknowledged, suggesting the possibility of their existence in acoustic models as well. By listening to dataset examples at different ranges of neuron activation we were able to uncover these polysemantic acoustic neurons. Initially, these neurons appeared to respond to a single phoneme when you only listen to the max activating dataset examples.  However, listening to examples at varying levels of activation reveals polysemantic behaviour. Presented in the following plots are the sounds that `neuron 1` and `neuron 3` in `blocks.2.mlp.1` activate on  at different ranges of activation. Additionally, audio samples are provided to illustrate this phenomenon.
<div style="display: flex; justify-content: center;">
    <img src="/blog/assets/images/whisper_interpretability/encoder/poly_ch_sh.png" alt="poly_ch_sh" style="width: auto; height: auto;" />
</div>
<div style="display: flex; flex-direction: column; align-items: center;">
    <img src="/blog/assets/images/whisper_interpretability/encoder/poly_c_g.png" alt="poly_c_g" style="width: auto; height: auto;" />
</div>



## Audio samples for polysemantic neurons

<details>
<summary>Neuron 1 ('sh/ch/j')</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp_1_1pkl_poly_sh.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp_1_1pkl_poly_ch.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp_1_1pkl_poly_j.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

<details>
<summary>Neuron 3 ('c/g')</summary>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp_1_3pkl_poly_c.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
<audio controls>
   <source src="/blog/assets/images/whisper_interpretability/audio/b2_mlp_1_3pkl_poly_g.wav" type="audio/wav">
   Your browser does not support the audio element.
</audio>
</details>

# The decoder is a weak LM
Whisper is trained exclusively on supervised speech-to-text data; the decoder is **not** pre-trained on text. In spite of this, the model still acquires rudimentary language modeling capabilities. While this outcome isn't unexpected, the subsequent experiments that validate this phenomenon are quite interesting/amusing in themselves.


## Bigrams
If we just use 'padding' frames as the input of the encoder and 'prompt' the decoder we can recover bigram statistics. For example, at the start of transcription, the decoder is normally prompted with:\
`<|startoftranscript|><|en|><|transcribe|>`

Instead we set the 'prompt' to be:\
`<|startoftranscript|><|en|><|transcribe|> <our_prompt_token>`

This is analogous to telling the model that the first word in the transcription is `<our_prompt_token>`.  

Below we plot the top 20 most likely next tokens and their corresponding logit for a variety of prompts. We can see that when the model has no acoustic information it relys on learnt bigrams.

![very](/blog/assets/images/whisper_interpretability/decoder/prompt_images/very_prompt.png)
![traffic](/blog/assets/images/whisper_interpretability/decoder/prompt_images/traffic_prompt.png)
![Good](/blog/assets/images/whisper_interpretability/decoder/prompt_images/Good_prompt.png)

## Embedding space

Bigram statistics are often learnt by the token embedding layer in transformer language models. Additionally in LLMs, we observe semantically similar words clustered in embedding space. This phenomenon also holds for Whisper, but additionally we discover that words with **similar sounds** also exhibit proximity in the embedding space. To illustrate this, we choose specific words and then create a plot of the 20 nearest tokens based on their cosine similarity.\
\
'rug' is close in embedding space to lug, mug and tug. This is not very surprising of a speech-to-text model; if you *think* you hear the word 'rug', it is quite likely that the word was in fact lug or mug.
![rug](/blog/assets/images/whisper_interpretability/decoder/embedding_space/rug_embed.png)
Often tokens that are close in embedding space are a combination of rhyming words **and** semantically similar words:
![UK](/blog/assets/images/whisper_interpretability/decoder/embedding_space/UK_embed.png)
![duck](/blog/assets/images/whisper_interpretability/decoder/embedding_space/duck_embed.png)
![tea](/blog/assets/images/whisper_interpretability/decoder/embedding_space/tea_embed.png)

## Features in the decoder

Finally, we collected maximally activating dataset examples (using the same dataset of 10,000 2s audio clips) for the neuron basis of decoder `blocks.0.mlp.1`. We find that they often activate on semantically similar concepts, suggesting that a) the model is already activing on the word level by the first MLP layer and b) it has aquired rudimentary language modelling capabilities like a weak LLM. Below we show these maximally ativating dataset examples for some neurons in `decoder.blocks.0.mlp.1`. 

### Neuron 10
|Example idx | 0 | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  |9|
| ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transcript  | Better food. |  I'm a steak  | cheese  |  on a meal. |  this salad |  I've been eating |  for lunch | Rice | and after dinner | dinner |

## Neuron 12
|Example idx | 0 | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9 |
| ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transcript  | than 15 |  the 8th  | at 5  |  Amen Perth |  of 16th |  the 20th |  6th year | Until the 6th | 3,000 | Stay at 4th |

## Neuron 14
|Example idx | 0 | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  |
| ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transcript  | or throw |  to knock the rear  | or shaking  |  such knock |  I'll drill you |  was brushing |  Just throw | swept the | to struck |



