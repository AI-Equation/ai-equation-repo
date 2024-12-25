# List of Artificial Intelligence AI Equations Formulas Latex Code Tutorials

## Introduction

Artificial Intelligence and Generative AI are growing fields of research and industrial application. With more technical breakthroughs in large language model (LLM), reasoning, CoT, Text generation, Video Generation, the mathmetical fundamentals are also important to advance further research and development. In this repo, we are collecting Artificial Intelligence AI and Robot related Equations formulas, papers, term explanation and latex code. You can use this blog as a bookmark to help advance your research. If you are interested to explore more equations in AI, ML, Robotics, Math and physics, you can visit the [DeepNLP Equation Database and Search Engine](http://www.deepnlp.org/search/equation) and [AI Agent Marketplace Search](http://www.deepnlp.org/search/agent). 

Meanwhile, If you have your own equations that you would like to bookmark and share to others, you can also use the [Equation Latex Code Bookmark Workspace](http://www.deepnlp.org/workspace/detail) to Save your equations latex code and related explanations. The AI equations collections cover several sub-categories, including AI Models equations, FineTuning methods, Alignment Optimization techniques, etc.


## Table of Content
### AI Models Equations
#### Transformer Model
#### Diffusion Model
### FineTuning
#### LOW RANK ADAPTATION LORA
###  Alignment Optimization
#### RLHF Reinforcement Learning from Human Feedback
#### PPO Proximal Policy Optimization
#### DPO Direct Policy Optimization
#### KTO Kahneman Tversky Optimisation


## Detail List of AI Equations
### AI Models Equations
### Transformer Model

**Equation**
![Transformer Equations Latex DeepNLP](https://raw.githubusercontent.com/AI-Equation/ai-equation-repo/refs/heads/main/docs/equation_transformer_1.jpg)


**Latex**

```
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
```

**Related** 

[paper](https://arxiv.org/abs/1706.03762) <br>
[equation](http://www.deepnlp.org/equation/transformer) <br>


### Diffusion Model

Diffusion Model Forward Process

Diffusion Model Forward Process Reparameterization

Diffusion Model Reverse Process

Diffusion Model Variational Lower Bound

Diffusion Model Variational Lower Bound Loss

**Equation**
![Diffusion Equations Latex on DeepNLP](https://raw.githubusercontent.com/AI-Equation/ai-equation-repo/refs/heads/main/docs/equation_diffusion.jpg)


**Latex**

```
q(x_{t}|x_{t-1})=\mathcal{N}(x_{t};\sqrt{1-\beta_{t}}x_{t-1},\beta_{t}I) \\q(x_{1:T}|x_{0})=\prod_{t=1}^{T}q(x_{t}|x_{t-1})

x_{t}=\sqrt{\alpha_{t}}x_{t-1}+\sqrt{1-\alpha_{t}} \epsilon_{t-1}\\=\sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t}\alpha_{t-1}} \bar{\epsilon}_{t-2}\\=\text{...}\\=\sqrt{\bar{\alpha}_{t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\epsilon \\\alpha_{t}=1-\beta_{t}, \bar{\alpha}_{t}=\prod_{t=1}^{T}\alpha_{t}

p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \\p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))


\begin{aligned}- \log p_\theta(\mathbf{x}_0) &amp;\leq - \log p_\theta(\mathbf{x}_0) + D_\text{KL}(q(\mathbf{x}_{1:T}\vert\mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0) ) \\&amp;= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \Big] \\&amp;= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \Big] \\&amp;= \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\\text{Let }L_\text{VLB} &amp;= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \geq - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)\end{aligned}


                     \begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0 \\
\text{where } L_T &= D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\
L_t &= D_\text{KL}(q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \vert\mathbf{x}_{t+1})) \text{ for }1 \leq t \leq T-1 \\
L_0 &= - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
```


**Related**

[Diffusion models equations](http://www.deepnlp.org/blog/latex-code-for-diffusion-models-equations) <br>
[Diffusion model forward process](http://www.deepnlp.org/equation/diffusion-model-forward-process) <br>
[Diffusion model forward process reparameterization](http://www.deepnlp.org/equation/diffusion-model-forward-process-reparameterization) <br>
[Diffusion model reverse process](http://www.deepnlp.org/equation/diffusion-model-reverse-process) <br>
[Diffusion model variational lower bound](http://www.deepnlp.org/equation/diffusion-model-variational-lower-bound) <br>
[Diffusion model variational lower bound loss](https://www.deepnlp.org/equation/diffusion-model-variational-lower-bound-loss) <br>


#### FineTuning

##### LOW RANK ADAPTATION LORA

**Equation**
![LOW RANK ADAPTATION LORA Equation Latex on DeepNLP](https://raw.githubusercontent.com/AI-Equation/ai-equation-repo/refs/heads/main/docs/equation_lora.jpg)


**Latex**

```
W_{0} + \Delta W_{0} = W_{0} + BA, h=W_{0}x + \Delta W_{0}x = W_{0}x + BAx, \text{Initialization:} A \sim N(0, \sigma^{2}), B = 0

```


**Related**

[paper](https://arxiv.org/abs/2106.09685) <br>
[LORA equation](http://www.deepnlp.org/equation/low-rank-adaptation-lora) <br>

### Alignment Optimization

#### RLHF Reinforcement Learning from Human Feedback


**Equation**
![RLHF Equations Latex on DeepNLP](https://raw.githubusercontent.com/AI-Equation/ai-equation-repo/refs/heads/main/docs/equation_rlhf.jpg)


**Latex**

```
p^*(y_w \succ y_l|x) = \sigma(r^*(x,y_w) - r^*(x,y_l)) $$ $$
\mathcal{L}_R(r_\phi) = \mathbb{E}_{x,y_w,y_l \sim D}[- \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))] $$ $$
\mathbb{E}_{x \in D, y \in \pi_\theta} [r_\phi(x,y)] - \beta D_{\text{KL}}(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x))
```



**Related**

[Equation RLHF reinforcement learning from human feedback](http://www.deepnlp.org/equation/rlhf-reinforcement-learning-from-human-feedback) <br>
[Blog](https://huggingface.co/blog/rlhf) <br>

#### PPO Proximal Policy Optimization PPO


**Equation**
![PPO Equations Latex on DeepNLP](https://raw.githubusercontent.com/AI-Equation/ai-equation-repo/refs/heads/main/docs/equation_ppo.jpg)



**Latex**

```
L^{CLIP}(\theta)=E_{t}[\min(r_{t}(\theta))A_{t}, \text{clip}(r_{t}(\theta), 1-\epsilon,1+\epsilon)A_{t}]
```


**Related**

[Proximal policy optimization PPO Equation](http://www.deepnlp.org/equation/proximal-policy-optimization-ppo) <br>
[Blog](https://spinningup.openai.com/en/latest/algorithms/ppo.html) <br>


#### Direct Policy Optimization DPO


**Equation**
![DPO Equations Latex on DeepNLP](https://raw.githubusercontent.com/AI-Equation/ai-equation-repo/refs/heads/main/docs/equation_dpo.jpg)


**Latex Code**

```
\pi_{r} (y|x) = \frac{1}{Z(x)} \pi_{ref} (y|x) \exp(\frac{1}{\beta} r(x,y) ) ,r(x,y) = \beta \log \frac{\pi_{r} (y|x)}{\pi_{ref} (y|x)} + \beta \log Z(x) ,p^{*}(y_{1} &gt; y_{2} |x) = \frac{1}{1+\exp{(\beta \frac{\pi^{*} (y_{2}|x)}{\pi_{ref} (y_{2}|x)} - \beta \frac{\pi^{*} (y_{1}|x)}{\pi_{ref} (y_{1}|x)}&nbsp; )}} ,\mathcal{L}_{DPO}(\pi_{\theta};\pi_{ref}) = -\mathbb{E}_{(x, y_{w},y_{l}) \sim D } [\log \sigma (\beta \frac{\pi_{\theta} (y_{w}|x)}{\pi_{ref} (y_{w}|x)} - \beta \frac{\pi_{\theta} (y_{l}|x)}{\pi_{ref} (y_{l}|x)} )] ,\nabla \mathcal{L}_{DPO}(\pi_{\theta};\pi_{ref}) = - \beta \mathbb{E}_{(x, y_{w},y_{l}) \sim D } [ \sigma ( \hat{r}_{\theta} (x, y_{l}) - \hat{r}_{\theta} (x, y_{w})) [\nabla_{\theta} \log \pi (y_{w}|x) - \nabla_{\theta} \log \pi (y_{l}|x) ] ] ,\hat{r}_{\theta} (x, y) = \beta \log (\frac{\pi_{\theta} (y|x)}{\pi_{ref} (y|x)})
```


**Related**

[equation](http://www.deepnlp.org/equation/direct-policy-optimization-dpo) <br>
[paper](https://arxiv.org/abs/2305.18290) <br>


#### KTO Kahneman Tversky Optimisation


**Equation**
![KTO Equations Latex on DeepNLP](https://raw.githubusercontent.com/AI-Equation/ai-equation-repo/refs/heads/main/docs/equation_kto_1.jpg)


**Latex**

```
f(\pi_\theta, \pi_\text{ref}) =&nbsp; \mathbb{E}_{x,y\sim\mathcal{D}}[ a_{x,y} v(r_\theta(x,y) - \mathbb{E}_{Q}[r_\theta(x, y')])] + C_\mathcal{D}
```

**Related**

[KTO Equation](http://www.deepnlp.org/equation/kto-kahneman-tversky-optimisation-equation) <br>
[paper](https://arxiv.org/abs/2402.01306) <br>



## References
#### Equation
[DeepNLP Equation Database](http://deepnlp.org/equation)
[Math Equations](http://deepnlp.org/equation/category/math)
[Physics Equations](http://deepnlp.org/equation/category/physics)
[NLP Equations](http://deepnlp.org/equation/category/nlp)
[Machine Learning Equations](http://deepnlp.org/equation/category/machine%20learning)
#### AI Agent Marketplace and Search
[AI Agent Marketplace and Search](http://www.deepnlp.org/search/agent) <br>
[Robot Search](http://www.deepnlp.org/search/robot) <br>
[Equation and Academic search](http://www.deepnlp.org/search/equation) <br>
[AI & Robot Comprehensive Search](http://www.deepnlp.org/search) <br>
[AI & Robot Question](http://www.deepnlp.org/question) <br>
[AI & Robot Community](http://www.deepnlp.org/community) <br>
##### AI Agent
[AI Agent Reviews](http://www.deepnlp.org/store/ai-agent) <br>
[AI Agent Publisher](http://www.deepnlp.org/store/pub?category=ai-agent) <br>
[Microsoft AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-microsoft-ai-agent) <br>
[Claude AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-claude-ai-agent) <br>
[OpenAI AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-openai-ai-agent) <br>
[AgentGPT AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-agentgpt) <br>
[Saleforce AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-salesforce-ai-agent) <br>
[AI Agent Builder Reviews](http://www.deepnlp.org/store/ai-agent/ai-agent-builder) <br>
##### AI Reasoning Chatbot
[OpenAI o1 Reviews](http://www.deepnlp.org/store/pub/pub-openai-o1) <br>
[OpenAI o3 Reviews](http://www.deepnlp.org/store/pub/pub-openai-o3) <br>
##### AI Video Generation
[Sora Reviews](http://www.deepnlp.org/store/pub/pub-sora) <br>
[Kling AI Reviews](http://www.deepnlp.org/store/pub/pub-kling-kwai) <br>
[Dreamina AI Reviews](http://www.deepnlp.org/store/pub/pub-dreamina-douyin) <br>
[Best AI Apps Review](http://www.deepnlp.org/store/pub) <br>
[AI Video Generator](http://www.deepnlp.org/store/video-generator) <br>
[AI Image Generator](http://www.deepnlp.org/store/image-generator) <br>
[AI Glasses Review](http://www.deepnlp.org/store/ai-glasses) <br>
[VR Glasses Review](http://www.deepnlp.org/store/vr-glasses) <br>
##### Robotics
[Tesla Cybercab Robotaxi](http://www.deepnlp.org/store/pub/pub-tesla-cybercab) <br>
[Tesla Optimus](http://www.deepnlp.org/store/pub/pub-tesla-optimus) <br>
[Figure AI](http://www.deepnlp.org/store/pub/pub-figure-ai) <br>
