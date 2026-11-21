# How Particle Sliders for Tiny Aya learns

The preference is written in language during training and controlled by a number
during inference. Here the preference is sympathetic, constructive support with
independent judgment. The ordinary question stays the same. The base model stays
frozen; a small, input-dependent adapter learns to move its internal response
toward states that successful instructions already elicit from that base.

This guide describes the **released Tiny Aya checkpoint**, not every possible
ParticleGAN recipe. Its saved configuration is `bispf_optimized` with
`loss="non_saturating"`, the pinned ParticleGAN source below, and Particle Sliders core **0.14.2**.
The saved recipe is authoritative: a generic description of Sinkhorn-relativistic
training or YuE2's training settings does not describe these weights.

## 1. Turn a preference into paired internal targets

Let $q_i$ be a training question, $p_i$ a successful behavioral instruction, and
$h_0$ the frozen model's last hidden vector just before the first answer token.
Define

$$
N_i=h_0(q_i),\qquad T_i=h_0(p_i,q_i),\qquad
S_{\theta,P}(i)=h_{\theta,P}(q_i;s=1).
$$

$N_i$ is the neutral state, $T_i$ the frozen positive target, and $S$ the adapted
student. Each has 2,048 coordinates. Both contexts contain the identical user
question and empty continuation history. Only the positive context receives the
behavioral instruction; the model's normal chat preamble remains in both.

For each of 64 TRAIN questions, the base generated an OFF answer and answers
under three candidate instructions. Jev scored the complete answers. A target
qualified only at behavior score at least 0.75 and strictly above matched OFF;
the highest-scoring qualifying instruction supplied that question's target.
This produced **37 contexts for 37 distinct questions**. Instructions can differ
between examples. DEV and confirmation questions supplied no training targets.

The generated answers establish which contexts work; their tokens are not the
training loss. Supervision covers the first-answer hidden state, rather than
every later answer position. Better full-answer behavior must therefore be
measured separately, and later role or reasoning errors can remain.

## 2. What the particle adapter computes

For a frozen attention projection $W_0$ and input column vector $x$, the branch
has a down projection $V$, router $R$, nonlinear bridge $\phi$, up projection
$U$, and shared learned particle table $P\in\mathbb R^{128\times4}$:

$$
\begin{aligned}
a&=Vx, & v&=R(a),\\
w&=\operatorname{softmax}(Pv/\sqrt{4}), & z&=P^\top w,\\
y&=W_0x+s\,\frac{\alpha}{r}\,U\phi([a;z]).
\end{aligned}
$$

The down/up maps have rank $r=8$, with $\alpha=8$, so $\alpha/r=1$.
The router has three width-16 hidden layers and returns four coordinates. The
bridge has three width-48 hidden layers and returns eight coordinates. Both use
LeakyReLU with slope 0.2. Their learned biases are included in $R$ and $\phi$;
the down/up maps have no bias. The frozen projection's existing bias, if any,
is included in the $W_0x$ shorthand.

Each of Tiny Aya's 36 layers receives Q, K, V and O branches: **144 branches**,
all sharing one 128-by-4 cloud. Their down/up maps, routers and bridges are
projection-specific. The input determines the softmax weights, so different
tokens can use different mixtures of the same learned particles. The table and
all branch parameters train together. At inference the table's values are fixed;
the input-dependent mixtures still change.

An ordinary LoRA has correction $BAx$, a fixed linear map that can be merged
into $W_0$. Here the routed correction $U\phi([Vx;z(x)])$ is nonlinear in $x$;
in general it cannot be merged into one fixed weight matrix. This is an
architectural distinction, not evidence of universal quality or efficiency
superiority over LoRA or SFT.

Strength $s$ multiplies the branch. **OFF ($s=0$) bypasses it exactly** in this
native implementation. Strength 1 is the trained positive endpoint; 0.25 and
0.5 are measured intermediate settings. Scaling internal corrections does not
linearly interpolate final text or guarantee monotonic improvement on each
question. Negative strengths have not been evaluated and are unsupported by
this backend. The released files require the custom native runtime, not a
standard PEFT loader.

## 3. Convert response error into a conditional GAN problem

ParticleGAN normally learns a generator together with a movable table of latent
particles. In this integration, the routed adapter is the generator and its
actual shared routing table is the participating particle bank. The Gaussian
reference below is a separate object; it is not a replacement for that bank.

For a positive response scale $a$, form

$$
e_i=\frac{S_{\theta,P}(i)-T_i}{a},\qquad
\epsilon_i\sim\mathcal N(0,\sigma^2I),\qquad
x_{r,i}=\epsilon_i,\qquad x_{f,i}=\epsilon_i+e_i.
$$

These weights used **$a=1$ and fixed $\sigma=1$**, without edit whitening or a
noise-annealing schedule. Real and fake share their Gaussian draw within each
pair. If the student matches its target, $e_i=0$ and the two references coincide.
Otherwise a critic can learn which directions reveal the response difference,
and gradients through that critic train the adapter.

The condition is the frozen neutral state plus context metadata:

$$
c_i=[N_i;\ L_{n,i}/512;\ L_{p,i}/512;\ H_i;\ s_i].
$$

$L_n$ and $L_p$ are neutral and positive token lengths; $H_i=0$ is the shared
history length and $s_i=1$ the training strength. This gives 2,052 condition
coordinates, not a learned question-ID embedding. The critic concatenates the
2,048-dimensional error reference and this condition, then applies
`Linear(4100,16) → SiLU → Linear(16,1)`. It outputs an unbounded raw logit.

## 4. The objective used by the released checkpoint

Write $r_i=D_\psi(x_{r,i},c_i)$, $f_i=D_\psi(x_{f,i},c_i)$, and
$\operatorname{sp}(u)=\log(1+e^u)$. The saved non-saturating loss is

$$
\begin{aligned}
\mathcal A_D&=\frac1B\sum_i[\operatorname{sp}(-r_i)+\operatorname{sp}(f_i)],\\
\mathcal A_G&=\frac1B\sum_i\operatorname{sp}(-f_i),\\
\mathcal L_D&=\mathcal A_D+\mathcal R_D,\\
\mathcal L_{\theta,P}&=\mathcal A_G+\mathcal R_P(P).
\end{aligned}
$$

$\mathcal R_D$ is the upstream BISPF critic penalty for an explicitly continuous
vector domain. $\mathcal R_P$ is the upstream SBNP particle regularizer; the saved
recipe has `prior_reg=0.01`. These symbols denote the actual versioned upstream
implementations, including their scaling and geometry, rather than substitute
WGAN or variance/covariance penalties. The source links below define their full
update laws. The scalar equations alone do not reproduce the optimizer.

The engine delegates to ParticleGAN's public factories:

```python
loss = recipe.make_loss()
penalty = recipe.make_critic_penalty(
    opt_d, collect_stats=True, input_domain="continuous",
)
spread = recipe.make_prior_regularizer(prior=prior)
```

Each completed update trains the critic with student responses detached, then
freezes the critic's parameters and trains the adapter and bank through its
input gradients. The two phases draw rows and Gaussian references independently.
The frozen base is outside the optimizers but remains differentiable with
respect to adapter outputs. No extra token cross-entropy, output MSE or SFT warm
start is added. ParticleGAN's `constraint_geometry_backward` and optimizer
factories retain the official update law.

The released winner completed **1,024 updates**, batch 8, with nominal adapter
rate **0.0000625**, critic **0.018**, and bank **0.03**. The saved optimizer is
DualNorm with its recorded dense profile. Fresh initialization preserves the
pretrained trunk and starts each up projection at zero; every resumed state is
preserved. Model-weight EMA is disabled for adapter, critic and bank. Training,
selection and inference use raw weights; upstream scalar optimizer statistics
are retained and are not model-weight averaging.

## 5. Measure the effect in finished answers

An internal-state game does not itself establish better responses. Let $O_i^s$
be the complete generated answer at strength $s$, and $I$ the frozen Jev behavior
instrument. Checkpoint/strength selection uses only

$$
G(s)=\frac1m\sum_{i=1}^m[I(O_i^s)-I(O_i^0)].
$$

Every question has equal weight. Hidden-state SSE is diagnostic only and never
selects or breaks ties. Catastrophic-output rejection is a separate qualified
check. A generation cut off by its safety cap remains unfinished and unscored.

Two learning-rate cohorts completed 1,024 updates each; 24 declared DEV cells
covered four checkpoints and three strengths per cohort. Their unique DEV
maximum was the lower-rate cohort at step 1,024, strength 1. That choice was
locked before the 64-question confirmation. All six confirmation arms and all
384 native-EOS answers are retained.

On those 64 fresh questions the chosen adapter averages **0.732656**, versus
**0.664180 OFF** and **0.697578 prompt-only**. These are model-judge scores, not
accuracy percentages. Sympathy judgment increases; the separate sycophancy
diagnostic does **not** improve over OFF. One training seed, synthetic related
scenario families and a shared curation/evaluation judge limit the result.
There is no matched behavior-SFT baseline or general superiority claim.

## Historical experiment sources and attribution

ParticleGAN and Particle Sliders predate this entry. The contribution here is
the Tiny Aya binding, preference curation, paired-state training/evaluation
pipeline, auditable experiment and complete-answer explorer.

- [ParticleGAN training source](https://github.com/HyperGAN/ParticleGAN/tree/1b8355b184b21bb7d81104ee590813c262f6df0a): official recipe, loss, penalties, regularizer and optimizers. Its released wheel SHA-256 is `c1c0de7a635ac1b1634ba158eed050b6f6aed993e01f9b463435430c77d846ae`.
- [Particle Sliders core used for the published run](https://github.com/HyperGAN/particle-sliders/tree/63f6e3942340d8aadca816800d19fb1ac88d92a9): the conditional engine, audio response adapter, routed projection and reference modules under `packages/particle-sliders-core/src/particle_sliders/`. Shared explanatory [math guide](https://github.com/HyperGAN/particle-sliders/blob/63f6e3942340d8aadca816800d19fb1ac88d92a9/docs/math.md); this checkpoint's saved non-saturating recipe, fixed scales and finished-output selection policy are specified above.
- [Tiny Aya model binding](https://github.com/HyperGAN/tiny-aya-particle-sliders/blob/bf0b1a7a8600efb0e814677e9a73314cbf9a60a3/src/evidence_dial/binding.py) and [behavior binding](https://github.com/HyperGAN/tiny-aya-particle-sliders/blob/bf0b1a7a8600efb0e814677e9a73314cbf9a60a3/src/evidence_dial/behavior_promptfix_binding_v4.py).
- [Winner record](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders/blob/main/winner.json), [raw checkpoint with saved recipe](https://huggingface.co/ntc-ai/tiny-aya-particle-sliders/blob/b7e7f256035b9eb8d4ded4d01af52effff2e51fc/adapter-raw.pt), and [data and audits](https://huggingface.co/datasets/ntc-ai/tiny-aya-particle-sliders/tree/main/promptfix37-h1024-v4).
- [Meta Glimmer paired-context reference](https://github.com/HyperGAN/meta-glimmer-particle-slider/tree/813ea28dd37c156a23470d1ef9a4f1e418c581ac) and [YuE2 model card](https://huggingface.co/ntc-ai/yue2-concept-sliders), the presentation example. YuE2's model, normalization, critic, noise schedule and weight-averaging settings are not used here.

Tiny Aya-derived weights and hidden states retain **CC-BY-NC-4.0**. Upstream
software retains its own licenses.
