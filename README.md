<div>
    <h1>RAG-ICL and DFPO: Enhancement methods for two learning paradigms in biomedical information extraction</h1>
</div>

<p align="center">
    <a 
        href=""><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-orange">
    </a>
    <a href="https://huggingface.co/Knifecat/DFPO-Gemma2">
        <img alt="Pretrained Models" src="https://img.shields.io/badge/🤗HuggingFace-Pretrained Models-green">
    </a>
</p>

<p align="justify">
    This repository provides code for the RAG-ICL and DFPO methods. RAG-ICL is based on the concept of Retrieval-Augmented Generation (RAG), optimizing in-context learning by selecting high-quality examples. <b>DFPO</b> (<b>D</b>ual-phase <b>F</b>ine-tuning and <b>P</b>reference <b>O</b>ptimization), on the other hand, uses a "two-stage fine-tuning and preference optimization" strategy, specifically designed for the data structure of biomedical information extraction tasks. It gradually strengthens the model's performance on positive samples and enhances overall effectiveness through preference optimization. Experimental results show that both methods achieve significant improvements over traditional approaches.
    <ul>
        <li>
            📖 Paper: <a href="">RAG-ICL and DFPO: Enhancement methods for two learning paradigms in biomedical information extraction</a>
        </li>
        <li>
            🤖 Model availiable: <a href="https://huggingface.co/Knifecat/DFPO-Gemma2">Knifecat/DFPO-Gemma2</a>
        </li>
        <li>
            📁 PreFt-Data and MFPEA: <a href="https://huggingface.co/Knifecat">Data</a>
        </li>
    </ul>
</p>
<h3>RAG-ICL Flowchart</h3>
<p align="center">
    <img src="assets/RAG-ICL.jpg" alt="RAG-ICL" width="50%">
</p>
<h3>DFPO Flowchart</h3>
<p align="center">
    <img src="assets/DFPO.png" alt="DFPO" width="60%">
</p>
<h3>DMCR Flowchart</h3>
<p align="center">
    <img src="assets/DMCR.png" alt="DMCR" width="50%">
</p>

<h2>Prerequisites</h2>
<ul>
    <li>Python 3.10 or higher</li>
    <li>PyTorch</li>
    <li>langchain</li>
    <li>transformers</li>
    <li>peft</li>
    <li>bitsandbytes</li>
    <li>accelerate</li>
    <li>deepspeed</li>
</ul>

Install the required packages using:
<pre><code>pip install -r requirements.txt</code></pre>

<h2>Runing</h2>
<h3>RAG-ICL</h3>
<p>RAG-ICL and assessment dataset generation</p>

```
python ./scripts/generate_data.py --all
```

<p>Filtering out high-quality examples</p>

```
python ./scripts/RAG-ICL/best_examples.py \
    --task "ner" \
    --one_dataset "bc5cdr" \
    --embedding_model_path /path/to/your/embedding_model \
    --rerank_model_path /path/to/your/rerank_model \
    --search_type "similarity" \
    --k 30 \
    --score_threshold 0.79 \
```

<h3>DFPO</h3>
<p>PreFt and Post-Ft</p>

```
accelerate launch ./scripts/DFPO/finetune.py \
    --modelpath /path/to/your/Base_Gemma2_model \
    --datapreprocess \
    --epoch 5 \
    --batchsize 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type "constant_with_warmup" \
    --learning_rate 2e-4 \
    --output_dir "../../result" \
```

<p>Preference Optimization</p>

```
accelerate launch ./scripts/DFPO/prefer_optimize.py \
    --modelpath /path/to/your/PreFt_Gemma2 \
    --beta 0.4 \
    --max_length 2048 \
    --max_prompt_length 512 \
    --learning_rate 2e-7 \
    --loss_type dpop \
    --dpop_lambda 2 \
    --output_dir "../../DFPO_result" \
```

<h2>Reminder</h2>
<p>
    The <a href="https://github.com/abacusai/smaug">DPOP</a> algorithm has not yet been integrated into Huggingface's trl library and needs to be implemented in DPOTrainer.
</p>
<p>The following is the implementation of TRL loss</p>

```
       pi_logratios = policy_chosen_logps - policy_rejected_logps
       logits = pi_logratios - ref_logratios
       penalty_term = torch.maximum(torch.zeros_like(policy_chosen_logps), reference_chosen_logps - policy_chosen_logps)
       logits += - self.lambda * penalty_term

            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
```






