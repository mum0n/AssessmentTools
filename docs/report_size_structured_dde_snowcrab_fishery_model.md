---
title:  'A multi-stage, delay differential model of snow crab population dynamics\
  in the Scotian Shelf of Atlantic Canada'
header: 'Choi 2024'
date: last-modified
date-format: "YYYY-MM-D"
# journal: 'biorxiv'
author:
- name: Jae S. Choi
  email: jae.choi@dfo-mpo.gc.ca
  # email: choi.jae.seok@gmail.com
  footnote: 1
  orcid: 0000-0003-3632-5723 
  corresponding: true
affiliation:
- number: 1
  name: Bedford Institute of Oceanography, Fisheriess and Oceans Canada
keyword: |
  Keywords - fishery assessment; snow crab; fishery footprint; Bayesian model; Julia; Turing; DifferentialEquations
abstract: |
  Snow crab (*Chionoecetes opilio*) are cold-water stenotherms in the northern hemisphere. As they are long-lived and have a complex life history, developing an operational model of population dynamics has been a challenge, especially in the context of an ever increasing and varied human footprint upon nature. Here we review past efforts at understanding the population dynamics of snow crab in an environmentally and spatiotemporally heterogeneous area, the Scotian Shelf of the northwest Atlantic of Canada. We address these difficulties with a moderately complex multi-stage, delay differential model and parameterize it leveraging Bayesian techniques. Operational solutions were stable and reasonable and permitted inference upon the intra-annual dynamics of snow crab. Further, a concept of a *Fisheries footprint*, akin to instantaneous fishing mortality rate, can be elucidated that directly addresses the conceptual impact of a fishery upon a non-stationary population. The approach is promising. The model suggests additional processes need to be accounted. We hypothesize that seasonal, interannual movement and spatiotemporally structured predation are key processes that require further attention. However, as computational costs are significant, these additional processes will need to be parameterized carefully. 
#standalone: true
citeproc: true
citeproc-method: biblatex
biblatex: true
biblatexoptions: |
 \usepackage[authordate, maxcitenames=1, uniquename=mininit, backend=biber, natbib]{biblatex-chicago}
# csl-refs: true
csl: media/chicago-author-date.csl
# https://www.overleaf.com/learn/latex/Biblatex_citation_styles
bibliography: references.bib
acknowledgements: |
additionalinformation: | 
# documentclasses: article, paper, cidarticle, REVTeX, scrartcl
documentclass: paper
papersize: letterpaper
fontsize: 12pt
toc: true
toc-depth: 4
number-sections: true
highlight-style: pygments
editor:
  render-on-save: false
  markdown:
    references:
      location: document
execute:
  echo: true
format:
  html: 
    code-fold: true
    code-overflow: wrap
    html-math-method: katex
    self-contained: true
    embed-resources: true
  pdf_document:
params:
  year_assessment: 2024
  model_variation: size_structured_dde_normalized
---

 

<!-- Preamble


This is a Markdown document ... To create HTML or PDF, etc, run: 

As it runs R and Julia, Quarto is probably the better tool to render. Alternatively, you can run everything directly and then render just the figures by turning off the R and Julia code chunks. To create an HTML via Quarto:

>> cd ~/projects/model_fishery/docs

>> make quarto FN=size_structured_dde_normalized YR=2024 SOURCE=~/projects/model_fishery/scripts WK=~/projects/model_fishery/outputs DOCEXTENSION=html PARAMS="-P year_assessment:2024 -P model_variation:size_structured_dde_normalized"

>> make quarto FN=size_structured_dde_normalized YR=2024 SOURCE=~/projects/model_fishery/scripts WK=~/projects/model_fishery/outputs DOCEXTENSION=html PARAMS="-P year_assessment:2024 -P model_variation:size_structured_dde_unnormalized"
 
  
Alter year and directories to reflect setup or copy Makefile and alter defaults to your needs.


# https://github.com/citation-style-language/styles  # to get more csl's
 
template related ...to get started:

Write paper in as *.md, refs in references.bib and use my_template.tex as a pandoc template
Modify makefile as required

citations: are cited as \cite{mittner2014brain} 

equations: 
  referenced as:  eq. (\ref{1}) 
  tagged as:  $$ \pi=\pi \tag{1a} \label{1}$$

pictures:
  referenced as: figure \ref{fig2}
    tagged as: ![my figure captions. \label{fig2}](Figures\Tolmukapea.jpg){width=100px height=50px} 
    
    no working: fignos ..  Fig. @fig:dummy{#fig:dummy width=40% height=20%}

tables: 
  do same as pictures or
  direct latex Tables: 
  
  \begin{table}[ht]
  \centering
  \caption{Probability to observe Bayes Factors of a certain magnitude or above for the used sample-size of $N=60$ assuming the original and the null-hypothesis.}
  \begin{tabular}{llrrr}
    & & \multicolumn{3}{l}{$P(\text{BF}\ge\theta)$}\\
    Hypothesis & BF Type & $\theta=3$ & $\theta=10$ & $\theta=20$ \\
    \hline
    $d\sim \mathcal{N}(1.57, 0.51)$ & JZS BF$_{10}$ & 0.98 & 0.97 & 0.96 \\
     & Replication BF$_{10}$ & 0.98 & 0.96 & 0.96 \\
     & Meta-Analysis BF$_{10}$ & 0.99 & 0.99 & 0.99 \\\cline{2-5}
    $d=0$ & JZS BF$_{01}$ & 0.81 & 0.00 & 0.00 \\
     & Replication BF$_{01}$ & 0.98 & 0.95 & 0.91 \\
     & Meta-Analysis BF$_{01}$ & 0.63 & 0.27 & 0.06 \\
     \hline
  \end{tabular}
  \label{tab:probbf}
  \end{table}

-->

# Introduction

Snow crab (*Chionoecetes opilio*) are large Crustaceans, exploited
primarily as a food source. They are also used as fertilizer, bait in
other fisheries and for the glucosamine polysaccharide derived from
chitin, known as chitosan. Chitosan is widely used in medicine as an
agent to slow bleeding from wounds
(\cite{Zhang_et_al_2015}, \cite{Moghadas_et_al_2016}), agriculturally as natural
fungicides and bactericides (\cite{Linden_Stoner_2007}), plastic replacement
(\cite{Tampieri_et_al_2003}) and even as a battery electrolyte
(\cite{Poosapati_et_al_2021}). In North America, including in the area of this
study, the Scotian Shelf of the northwest Atlantic of Canada
(Fig. \ref{figareamap}, the largest
of males (\> 95 mm carapace width) are preferentially captured due to
their higher meat-yield and consumer preferences of larger claws, a
trait that occurs only at the terminal molt to maturity. This
market-driven protection for the female reproductive crab which never
reaches such sizes and the smaller immature crab represents a form of
built-in protection for the population which spans 10 years of age or
more, from larval release; it helps to offset otherwise heavy
exploitation pressures of mature males by fishers with advanced
technological and historical knowledge of their environment and the
species' aggregation patterns. Currently, every known population of snow
crab is exploited by humans. As such, it is imperative that exploitation
occurs in a responsible and sustainable manner.

![The area of interest in the Scotian Shelf of the northwest Atlantic
Ocean, Canada. This area is at the confluence of the Gulf Stream from
the south and south east along the shelf edge, Labrador Current and St.
Lawrence outflow from the north and north east, as well as a nearshore
Nova Scotia current, running from the northeast. It is hydro-dynamically
complex due to mixing of cold, low salinity water from the north with
the warm saline water from the south. Shown also are the managed Crab
Fishing Areas divided by thick dashed lines: NENS (North-Eastern Nova
Scotia), SENS (South-Eastern Nova Scotia), CFA 4X (Crab Fishing Area
4X).\label{figareamap}](media/map.png){#figareamap width=70%}

Unfortunately, characterizing snow crab population dynamics is
remarkably difficult (Fig. \ref{figlifehistory}) because they are long lived, have a complex life
history, and large vertical and horizontal shifts in spatial
distributions as they grow older (ontogenetic shift in habitat).
Functionally, they participate in the pelagic and subsequently benthic
ecosystems and associated nutrient and carbon cycles. Some of the
notable life history features of snow crab that make them so interesting
but difficult to model, include: sexual dimorphism with mature males
being much larger than mature females; pelagic larval stages and benthic
pre-adolescent and adult stages; semi-annual and annual molts depending
upon size, age and environmental conditions; skipping moults if
conditions are poor; terminal molt to maturity; and longevity up to 15
years. They also have a narrow range of temperature preferences
(\cite{Foyle_et_al_1989}; \cite{Sainte-Marie_Lafrance_2002}; \cite{Kuhn_Choi_2011}). They
are thought to avoid temperatures above $7^{\circ}C$, as metabolic costs
have been shown to exceed metabolic gains above this temperature in
laboratory studies (\cite{Foyle_et_al_1989}). Smaller crab and females also
have differences in thermal preferenda (\cite{Choi_et_al_2022}). Further, snow
crab are generally observed on soft mud bottoms; with smaller-sized and
molting crabs showing preference for more complex (boulder, cobble)
substrates, presumably as they afford more shelter
(\cite{Sainte-Marie_Hazel_1992}; \cite{Comeau_et_al_1998}). This life history
complexity induces complexity (non-random structure) in their dynamics
and spatial distributions.

In the area of study, the continental shelf of the northwest Atlantic
Ocean (Fig. \ref{figareamap}), snow
crab are further exposed to high bottom temperature variability due to
the confluence of a number of oceanic currents: the warm Gulf Stream,
the cold Labrador current, low salinity outflow from the St. Lawrence
river and the cold coastal Nova Scotia current. In this region, snow
crab are generally observed between depths of 50 to 300 m and between
temperatures of $-1$ to $11^{\circ}C$ (\cite{Choi_2011}). As the focal area is
thermally complex, their spatial distributions can fluctuate seasonally
and annually. The additional factors of global rapid climate and
ecosystem change confounds make this understanding even futher.

![Life history patterns of snow crab and approximate timing of the main
life history stages since larval release of snow crab and size (measured
as carapace width; mm) and instar (Roman numerals). Size and timings are
specific to the area of study and likely vary with regional
environmental conditions, food availability and genetic variability.
Brooding time is variable and is between 1 and two years. Initiation of
terminal molt to maturity (orange and blue lines) also vary in timing.
After initiation, longevity is thought to be about 5 to 6 years (dashed
lines). Green solid line identifies approximate size and age of males
exploited by humans.\label{figlifehistory}](media/life_history.png){#figlifehistory  width=90%}

Snow crab eggs are brooded by their mothers for up to two years,
depending upon ambient temperatures as well as food availability (which
also depends upon temperature as it influences overall system primary
and secondary production) and the health condition and maturity status
of the mother (up to 27 months in primiparous females -- first breeding
event; and up to 24 months in multiparous females -- second and
subsequent breeding events; \cite{Sainte-Marie_1993}). More rapid development
of eggs, from 12 to 18 months, has been observed in other areas
(\cite{Elner_Beninger_1995}; \cite{Webb_et_al_2007}). Over 80% of the female snow
crab on the Scotian Shelf are estimated to follow an annual cycle,
possibly due to extended periods of exposure to warmer temperatures
(\cite{Kuhn_Choi_2011}). A primiparous female of approximately 58 mm carapace
width produces between 35,000 to 46,000 eggs, which are extruded between
February and April (\cite{Sainte-Marie_1993}). Multiparous females are thought
to be more fecund, with more than 100,000 eggs being produced by each
female. Eggs are hatched from April to June when larvae are released.
This pelagic larval stage lasts for three to five months (Zoea stages I
and II) during which snow crab feed upon zooplankton. Pelagic stages
seem to have highest abundance in October and so may begin settling in
January (\cite{Kuhn_Choi_2011}). Thereafter, they settle closer to the ocean
bottom in their Megalopea stage. Very little is known of survival rates
at these early life stages, but they are thought to depend highly upon
temperature (\cite{Sainte-Marie_Lafrance_2002}).

There have only been a small number of attempts at modeling the dynamics
of snow crab populations. Being long-lived, sexually dimorphic, pelagic
and benthic organisms, usually living far offshore, they are not easily
surveyed. Most population assessments approach abundance estimation with
a fishery- or survey-based relative index of a narrow segment of the
population (the fishable component). Further, aging of snow crab is not
possible due to a lack of retention of calcified body parts through
molts. This has encouraged adoption of size-based models
(\cite{Siddeek_et_al_2009}; \cite{Cadigan_et_al_2017}; \cite{Ianelli_et_al_2022}), often
with strong and questionable assumptions. For example,
(\cite{Siddeek_et_al_2009}) assume the male fishable component is the spawning
stock biomass, though of course it is the females that are the
reproductive component and completely unexploited with a shorter life
span. Similarly, (\cite{Choi_Zisserson_2011}) implicitly assumes that the
fishable biomass regenerates itself by application of a biomass dynamics
model. Importantly, sex and size-selectivity of sampling gear are almost
always assumed to be a constant or some smooth monotonic function of
body size, and so static across time and space or known without bias.
These are significant and biologically problematic assumptions. The
problems encountered are fundamental issues shared with all other
attempts at assessment. Specifically, they are the interplay between
snow crab life history and behavior; our inability to observe them
without bias due to sampling design being non-random with respects to
the environmental factors controlling their distribution and abundance;
and the sampling machinery (mesh size, trawls that cannot access rugose
areas) that can only observe a small fraction of the population.

This paper identifies a novel approach that overcomes some of these
difficulties in modelling snow crab, in particular, by incorporating
observation error across sex and stage-structure (often called a
*latent* or *state-space* model) and non-stationary habitat variability
to simultaneously model population dynamics and infer population
parameters. We examine its utility and demonstrate the mechanism by
which we can make these and more complex models operationally viable
through the use of efficient and modular computing that simultaneously
models and infers parameters by leveraging the **Julia** programming
environment (\cite{Bezanson_et_al_2017}) and the supporting **Turing** library
(\cite{Ge_et_al_2018}) for Bayesian parameter inference and the
**DifferentialEquations** library (\cite{Rackauckas_Nie_2017}) for dynamical
modeling. This examination is conducted in the Scotian Shelf Ecosystem
(Fig. \ref{figareamap}) where the
variability of ocean climate is known to be high
(\cite{Choi_et_al_2004}; \cite{Drinkwater_2005}) and where we also have a
consistently sampled population since the late 1990s. The principal
conclusion of this study is that the novel proposed model is informative
and demonstrates utility in understanding snow crab dynamics,
integrating all available information and versatility in
incorporating/aggregating ecosystem effects through its inlfuence upon
habitat viability. It is sufficiently flexible in approach to permit
other ecosystem factors such as classical inter-specific interactions
and movement, which are planned for future studies.

# Methods

Data collection and subsequent index estimation are described in
(\cite{Choi_2020}; \cite{Choi_2022}; \cite{Choi_et_al_2022}). In short, sampling in an
unbiased manner under complex hydrodynamic conditions is particularly
difficult as the random (spatially) stratified sampling that is usually
adopted to account for such variability, fail to do so, due to the large
dynamic (i.e.,spatiotemporal) structures operating on scales comparable
to the domain. A model-based approach (*Conditional AutoRegressive
Spatiotemporal Models*, **CARSTM**, <https://github.com/jae0/carstm>), a
simple extension of Generalized Linear Models that accounts for
spatiotemporal autocorrelated random effects in a Bayesian context and
computed with INLA (\cite{Rue_et_al_2016}), are used to address the estimation
problem. More specifically, biases induced by differential
spatiotemporal variability in habitat for differing life stages which,
upon aggregation across space, permits an informative index of abundance
comparable across time (years). This study focuses upon the indices of
abundance derived from this process for the period from 1999 to 2022.
Importantly, no survey was conducted in 2020 due to Covid-19 related
uncertainties. The index generation method being Bayesian, imputes these
estimates, but variability attached to this year is large. These
aggregate timeseries data and supporting **Julia** models used in this
paper are available at:
[model_fishery](https://bitbucket.org/autocatalysis/model_fishery).

The well understood *phenomenological* single component model of
Verhulst (\cite{Verhulst_1845}; \cite{Bacaër_2011}) is the usual starting point for
most models and has been used to model snow crab for many years
(\cite{Choi_Zisserson_2011}; \cite{Smith_et_al_2012}):

$${\frac{\mathit{dN}}{\mathit{dt}}_{}=\mathit{rN}}{({1-\frac{N}{K}})}. \tag{eq. 1} \label{eq1}$$

It is parameterized by an intrinsic rate of increase ($r$) representing
the maximum exponential rate of increase and carrying capacity ($K$) the
upper threshold. When numerical abundance $N\rightarrow0$, the loss term
also $\frac{N}{K}\rightarrow0$, and so
$\frac{\mathit{dN}}{\mathit{dt}}_{}\rightarrow\mathit{rN}$. When
$N\rightarrow K$, then $\frac{N}{K}\rightarrow1$ and so the loss rate
approaches $\mathit{rN}$ and the overall rate of change approaches zero
${\frac{\mathit{dN}}{\mathit{dt}}_{}\rightarrow0}{}$. Parenthetically,
this can be simplified further by dividing both sides by $K$ to give:
${{\frac{\mathit{dN}}{\mathit{dt}}/K}_{}=\mathit{rN}}{{({{1-r}\frac{N}{K}})}/K}.$
By focusing upon $n=\frac{N}{K}$ as a non-dimensional number ranging
from (0,1), this becomes:
${\frac{\mathit{dn}}{\mathit{dt}}_{}=\mathit{rn}}{({1-n})}$. For
parameter estimation/inference, this form is useful as it is simpler and
the magnitudes of variables are mostly similar, with the exception of
$K$. This renders beneficial properties to numerical optimization
procedures. The sigmoid nature of the equation has rendered it a
frequently encountered model that shows versatility in
phenomenologically describing population dynamics. Its discrete form is
particularly common in fisheries settings
(\cite{Schaefer_1954}; \cite{Meyer_Millar_1999}) and has also been used to describe
snow crab dynamics in the area of study (\cite{Choi_Zisserson_2011}). The
focus is usually upon biomass $B$ and its normalized value,
${{b=B}K^{-1}}_{},$ rather than numerical abundance:
${b_{t+1}={b_{t}+\Delta}}b_{t},$ where
$\Delta b_{t} = rb_{t}{({1-b_{t}})}-\mathit{Fishing}_{t}K^{-1}.$ Here,
$\mathit{Fishing}_{t}$ is scaled by $K$ to make it non-dimensional and
also in the interval (0,1); it is usually treated as an external factor
(perturbation), measured without error. Of course, there is usually
observation error and/or illegal removals that will get erroneously
absorbed as a biological process; in this case by elevating the
intrinsic rate of increase to compensate for the additional losses. This
will have an effect of creating bias and uncertainty in related
parameters (see below). Note also that the discretization to an annual
basis is a significant assumption as the time span is large relative to
the processes of most biota. This has the effect of averaging out
sub-annual dynamics. This means temporal aliasing or discretization
errors and censoring are introduced which ultimately increases process
and observation errors (see subannual dynamics, below).

From this *phenomenological* view, the minimal parameters required to
estimate biological reference points, and the relative distance a system
is from such reference points help delimit the status of a population
(\cite{Schaefer_1954}; \cite{Meyer_Millar_1999}). Values such as Maximum sustainable
yield $\left({\mathit{MSY}={\mathit{rK}/4}}\right)$ and the fishing
mortality associated with such a yield
$\left({\mathit{FMSY}={r/2}}\right)$ are commonly used to help define
some consistent landmarks of scale for use as management reference
points to guide the implementation of a consistent *Precautionary
Approach* for fishery exploitation (\cite{Smith_et_al_2012}).

Many approaches exist to estimate these model parameters. Currently,
*Maximum Likelihood* approaches dominate due to their computational
speed. However, it has been the author's experience with this data that
they do not navigate and optimize very high-dimensional parameter spaces
reliably, especially in the delay differential equation models that we
explore, below. This renders their utility in an operational setting,
minimally useful. The related *Maximum A-Posteriori* solutions, where
using the same optimization techniques but with the addition of
"prior-like" constraints can result in marginally more stable results,
but they still have tremendous difficulty with high dimensional
parameter spaces and the associated numerous local optima/multiple
equilibria.

Here we use Bayesian inference, as informative priors for these
parameters are explicit and variance propagation of index variables can
be accomplished by using them as priors to respective error terms. They
facilitate parameter estimates that are stable and credible, given some
prior knowledge of latent processes and observation models
(\cite{Meyer_Millar_1999}). Previously, JAGS (Just Another Gibbs Sampler
(\cite{Plummer_2003})) and STAN (\cite{Stan_2022}) were used to compute the
posteriors via MCMC (\cite{Choi_Zisserson_2011}). The latter uses the more
efficient NUTS sampler which significantly speeds up the estimation
process of these discrete difference equation models. Presently, we use
**Julia** programming environment (\cite{Bezanson_et_al_2017}) and the
supporting **Turing** library (\cite{Ge_et_al_2018}) for parameter inference
and the **DifferentialEquations** library (\cite{Rackauckas_Nie_2017}) for
modeling delay differential equation models; they demonstrate efficiency
and flexibility to simultaneously model and infer parameter values in
such problems, relative to basic MCMC procedures due to use of automatic
differentiation and heavily optimized solution engines.

Specifically, the latent ("real" but unobserved) biomass was assumed to
have a *process model error* that is a recursive Gaussian or Normal
distribution ($N$) with a standard deviation $\sigma_{p}$ (Bolker 2008)
such that: 

$$b_{t+1}\sim N\left({{{b_{t}+r}b_{t}}{{({1-b_{t}})}-\mathit{Fishing}}_{t}K^{-1},\sigma_{p}}\right). \tag{eq. 2} \label{eq2}$$

Here, "$\sim$" indicates "distributed as". Marginally informative priors
were assumed: $r\sim N\left({1.0,0.1}\right)$, and
$K\sim N\left({\kappa,0.1\cdot\kappa}\right)$. The prior mean of the
carrying capacity ( $\kappa={\lbrack{5.0,60,1.25}\rbrack}$, in kt, for
the NENS, SENS, and CFA 4X, respectively;
Fig. \ref{figareamap}) were based
upon historical analyses with other analytical procedures, namely
Geostatistical Kriging with External Drift and Generalized Additive
Models (\cite{Choi_et_al_2005b}; \cite{Choi_2011}).

It is further assumed that the real unobservable (latent) process $b$ is
sampled with error (observation standard deviation $\sigma_{o}$ ). The
*observation error model* in most fishery applications is a simple
multiplicative factor, often referred to as a "catchability" coefficient
$q$. In this case we assume that: $q\sim N\left({1.0,0.1}\right)$, and
so the *observation error model* becomes:

$$Y_{t}\sim N\left({q\mathit{Kb}_{t},\sigma_{o}}\right). \tag{eq. 3} \label{eq3}$$

This means that the observed biomass indices $Y$ are some constant
fraction of the true abundance $b_{t},$ across all sampling time (years,
season) and locations. The recursive logistic process model
(\ref{eq2}) in tandem
with the observation model (\ref{eq3}), we will refer to it as, "*Model 1*".

![A graphical representation of *Model 1* (simple logistic; \ref{eq1}).
Here, ${b=B}K^{-1}$ is a non-dimensional number that ranges from (0, 1)
that represents the biomass $B$ after being scaled to $K$. The loop
$\mathit{rb}$ identifies the growth rate and the loss term $rb^{2}$
identifies the quadratic increase in "mortality" as $b\rightarrow1$.
Fishing is also scaled to $K$, such that:
${\frac{\mathit{db}}{\mathit{dt}}_{}=\mathit{rb}}-\mathit{rb}\cdot{{b-f}={\mathit{birth}-\mathit{death}-\mathit{fishing}}}$.
As such, $rb2$ represents the non-fishing related mortality that is
density dependent. \label{figlogisticdiag}](media/logistic_diagram.png){#figlogisticdiag width=30%}

From the graphical representation of the processes of *Model 1* (Fig. \ref{figlogisticdiag}), 
one can immediately see that there is no input term from outside of
the system: it is a simple *phenomenological* self-loop model with two
outputs (death and fishing). The term $\mathit{rb}$ can be referred to
as the (net) "birth process"
(\cite{Gillespie_1992}; \cite{Qian_Bishop_2010}; \cite{Albert_2016}), and represents a
first order processes that increases $b$ (via birth, growth, enhanced
survival, improved food availability, movement, strong year class
strength, etc.), through the single (static) parameter $r$. The
mechanism is not identified and is implied to be some internal recycling
of $b$. In reality, there are biological mechanisms involved and
associated parameters that are not-static and non-stationary (in first
and second order; across time and space). Growth in biomass or increase
in numbers is seasonal and pulsed, due to the molting process and body
mass increases, that progress at variable rates depending upon time of
year and environmental variability. Across years, there are strong and
weak year classes due to match-mismatch type processes
(\cite{Cushing_1969}; \cite{Durant_et_al_2007}), and so also pulsed. These simple
models cannot address these more realistic dynamics and so we obtain
instead, an average state, with poor resolution of peaks and troughs and
associated parameter estimates that are likely to be in error.

The strength of *Model 1* is simplicity; but this simplicity is also a
weakness. As the biomass dynamics model is phenomenological; it is
heuristic, without mechanism. For example, fishable biomass "creates"
fishable biomass at a maximum exponential rate (r) to a limit of K. In
reality, fishable biomass being mature male snow crab, does not produce
more mature male snow crab; instead, it is the female mature population
that creates the eggs and broods them to larvae, some fraction of which
eventually mature into the fishable component. The dynamics and
longevity of these females are very different from those of the males,
often showing unbalanced sex ratios and differential utilization of
habitat and of course complete unfished (\cite{Choi_et_al_2022}). The rate of
increase in biomass is not a simple exponential rate; instead, a certain
number of fishable crab die from various causes (predation, disease,
competitive fighting), a certain number of recruits terminally molt to a
size in the fishable range to increase their numbers, they grow in
weight as they eat, some fraction move in and out of a given area, and
some are fished; these rates are not constant in time nor space and
there are many lags in time. Fundamentally it is a numerical process. So
*Model 1* from the perspective of "biomass dynamics" is less than
satisfactory in terms of biological realism. However, in the heuristic
perspective, it implies that the biomass in one year is related to the
biomass in the previous year within some constraints, constraints that
are only diffusely/indirectly related to the "real" mechanistic
processes represented by individual-individual interactions. As such,
they can be seen as a temporal autoregressive model.

These constraints unfortunately result in a model that is sometimes not
responsive enough to large fluctuations in dynamics caused by intrinsic
and/or extrinsic factors. For example, due to the pulse-like dynamics
observed since the late 1990s, no recruits entered fishable components
for a number of years, even though their abundance was low (\cite{Choi_2011}).
*Model 1* expected recruitment simply because biomass was low. This was
an erroneous expectation due to the low numbers of pre-recruits that
lasted for an extended period of time; the extreme warm bottom
temperatures during this period and associated shifts in their spatial
distributions; the increase in abundance of predators that followed
which potentially reduced the strength of that recruitment. *Model 1* is
too simple to express the expectations of such highly nonlinear
pulse-like dynamics and interactions with environment and predators. In
another example, there was an extreme warming event in 2021 that
significantly altered and constricted the spatial distribution of snow
crab (\cite{Choi_et_al_2022}). Again a simple model such as *Model 1*, with
static expectations of environmental conditions (*stationarity*) are not
able to account for such effects.

In an attempt to *begin* addressing these issues, we entertain *Model 2*
(Fig. \ref{figdddiag}) which
resolves some size groups, sex and maturity and permits time-varying
viable habitat area (*non-stationarity*). *Model 2* is, therefore, an
intermediately complex, marginally more structured and mechanistic
(*ontological*) model relative to *Model 1*.

![Graphical structure of the six-component *Model 2* of snow crab with
associated numerical flows. $F$ indicates mature females, determined
from size range and morphometric maturity. Associated with each
component is a carrying capacity ($K_{i}$), and a death rate $w_{i},$
and molt rates $v_{\mathit{ij}}$ ; and birth/survival rates $b_{i}$.
$F_{t-8}$ is the state of mature females $F$ from 8 years prior. Fishing
is considered an external, deterministic sink imposed by external
factors. \label{figdddiag}](media/delay_differential_diagram.png){#figdddiag
width=60%}

There is approximately an 8+ year period that is required for females
$F_{t-8}$ to produce the next generation of instar 9 females and males,
denoted by $F_{t}$ and $M_{5,t}$, respectively. They represent crabs in
the range of approximately 40 to 60 mm carapace width. The "birth" rates
represent a combination of egg production and larval survival to instar
9+. If the population is stable, "birth" rates are expected to be
similar in magnitude to overall death rates. Note that "stage" is
defined based upon size-sex-morphometric traits and so misclassification
is likely. The error in such knife-edge stage determinations can be
substantial if growth and maturity schedules vary significantly. It is
assumed, here, that in the aggregate, this has been stable in the survey
record (1999 to 2022). This may of course be incorrect if persistent
size/stage/age related mortality occurs due to exploitation/predation
patterns and/or directional environmental change such as climate warming
or food availability shifts.

Some fraction of instar 9 males transition to instar 10 ($M_{4}$), and
then 11 ($M_{3}$) to 12 ($M_{2}$) and 13+ ($M_{1}$); each transition
rate parameterized by $v$, with numeric indices identifying the instar
pairs. Due to the knife edge-cut of stages, there will be
misclassification issues which will be absorbed by these transition
rates. For example, a small fraction of instar 11 crab ($M_{3}$) will be
large enough to be considered a fishable size (\>95mm carapace width);
but most will molt into instar 12 ($M_{2}$); this would result in a
reduced molt transition rate. $M_{2}$ will represent a composite group
of most crab that have entered fishable size but are still immature.
Some fraction of this group will mature into the fishable component in
the next year; the fractional nature will result in a smaller molt
transition rate. Others will continue to molt to instar 13 and higher;
for our purposes, all morphologically mature males of fishable size will
be considered $M_{1}$ and so it represents a range of different ages.
This is especially the case as terminally molted crab can live for up to
another 5 years; this aggregation will effectively decrease mortality
rate of the group.

Survey sampling and estimation of instar 9 to 13+ (40 mm to 130+ mm
carapace width) is reasonably informative. Earlier instars tend to have
divergent habitat preferences relative to those of the fishable
component and are known to be poorly/erratically sampled due to size
approaching mesh size of sampling nets. This is because the snow crab
surveys are primarily optimized for sampling the fishable component (\>
95 mm carapace width), both in terms of mesh size and choices of
sampling location. The utility of these additional compartments is that
they also permit more stable parameter estimation and forward
projections that are biologically more reasonable (mechanistic). The
challenge is to keep track of much more information due to the increased
realism and computational limits. Each state variable
$U={({M_{1},M_{2},M_{3},M_{4},M_{5},F})}$, is scaled by their respective
carrying capacity $K_{.}$, such that they are non-dimensional numbers:
${u_{i}={({U_{1}K_{1}^{-1},U_{2}K_{2}^{-1},U_{3}K_{3}^{-1},U_{4}K_{4}^{-1},U_{5}K_{5}^{-1},U_{6}K_{6}^{-1}})}}.$

Thus, for example, in the molt transition process from $i=2$ to $j=1$,
the instantaneous rate is:

\begin{align*}
  v_{21}u_{2,{t-1}}{a_{21}} &= v_{21}{({U_{2,{t-1}}K_{2}^{-1}})}K_{2}K_{1}^{-1} \\
    &= v_{21}U_{2,{t-1}}K_{1}^{-1}.
\end{align*}

The multiplier ${a_{21}=K_{2}}K_{1}^{-1}$ is required to convert
normalized density of category 2 to that of category 1 via the ratio of
the respective normalizing constants, $K_{i}.$ As with *Model 1*, first
order birth/survival rates $b_{i}$ are assumed: ${\beta_{i}=b_{i}}u_{i}.$

So far, most of the core model has been the same as *Model 1*, only
applied to more components with some appropriate time lags. Where *Model
2* diverges from *Model 1* is by decomposing mortality into two
components: the first component is a simple background mortality
parameterized as a first order decay ${({\omega_{i}u_{i,t}})};$ the
second component is mortality associated with fluctuations of viable
habitat, parameterized as in the logistic equation as in *Model 1* as a
second order process
$\psi_{i}\frac{U_{i,t}}{K_{i,t}H_{\mathit{it}}}\cdot U_{i,t}$. As such,
it incorporates all density dependnent processes such as competition and
interaction between stages. Note the congruence of this term with that
of the $r\frac{N}{K}\cdot N$ loss term of *Model 1* (see Fig. \ref{figdddiag}, \ref{eq1}).
With normalization to $u$, we obtain:

$${w_{i,t}=\omega_{i}}{u_{i,t}+\psi_{i}}\frac{{u}_{i,t}}{H_{i,t}}\cdot{u}_{i,t}.$$

Here, ${H={V/\mathit{\max}}}{(V)},$ represents the viable habitat
surface area $V{}$ normalized to a maximum value of 1, where the maximum
is defined in the reference/focal time period. This modifies the
carrying capacity proportionately and so $\mathit{KH}$ can be seen as
the "effective carrying capacity", adjusting for viable habitat
fluctuations. The assumption here is that when viable habitat area
declines by some fraction $H$, so too does the effective carrying
capacity, proportionately.

The inclusion of predator-prey and other ecological processes are
therefore straightforward extensions as additional terms in this type of
model formalism are well understood
(\cite{Lotka_1920}; \cite{Bertalanffy_1950}; \cite{Pianka_1970}; \cite{Wangersky_1978}). Such
extensions are planned for future research and evaluation.

The full set of delay differential equations are, therefore:

$$\begin{aligned}
\frac{\mathrm{d} u_{1}}{\mathrm{d} t} &= v_{21}u_{2,{t-1}}a_{21}-w_{1,t}-\mathit{Landings}_{t}K_{1,t}^{-1} \\ 
\frac{\mathrm{d} u_{2}}{\mathrm{d} t} &= v_{32}u_{3,{t-1}}a_{32}-v_{21}u_{2,t-1}-w_{2,t} \\
\frac{\mathrm{d} u_{3}}{\mathrm{d} t} &= v_{43}u_{4,t-1}a_{43}-v_{32}u_{3,t-1}-w_{3,t} \\
\frac{\mathrm{d} u_{4}}{\mathrm{d} t} &= v_{54}u_{5,t-1}a_{54}-v_{43}u_{4,t-1}-w_{4,t} \\
\frac{\mathrm{d} u_{5}}{\mathrm{d} t} &= b_{5}u_{6,t-8}a_{56}-v_{54}u_{5,t-1}-w_{5,t} \\
\frac{\mathrm{d} u_{6}}{\mathrm{d} t} &= b_{6}u_{6,t-8}-w_{6,t}.
\end{aligned} \tag{eq. 4} \label{eq4}$$

*Model 2* (\ref{eq4}) is, therefore, a more mechanistic (structured) extension of
*Model 1* that brings to bear additional information available about the
abundance of other size and sex classes and transition rates between
them, and mortality rates that depend upon variations of numerical
density in viable habitat surface area.

Fishery landings were converted to number based upon annual average
weight of the fishable component estimated from surveys and discretized
to a two week time step within each year. The latter was to avoid strong
discontinuities which facilitates differential equation modeling and
parameter inference. Survey-based indices of numerical abundance were
estimated using a Hurdle process, where probability of observation and
the numerical abundance of positive-valued observations were modeled as
an extension of small-area analysis (Conditional AutoRegressive Models
in space and time; see methods in \cite{Choi_et_al_2022}; \cite{Choi_2022}).
Modeled average weights, also analyzed using the same small-area
space-time analyses were used to convert numbers back to biomass to make
comparisons with *Model 1*.

In *Model 1*, the meaning of $q$ is clear: a survey catches only some
(fixed) fraction of the true abundance. This is a reasonable first
approximation. However, what is implied is: first, this fraction is
unchanging, applying equally in high abundance and low abundance years
and areas, whether they are in the mating or molting season or not.
Ignored are issues such as differential net saturation, aggregation
behavior, even when bottom temperatures can force aggregation and
dispersal away from or into other areas. The second issue implied by $q$
is that when a survey fails to capture snow crab, that there truly is no
snow crab in the area: an observation of zero is a true zero. ***This is
clearly false***. Visual observation of trawl operation through video
monitoring have shown that crabs are missed: when they are in a slight
depression, when they are close to protruding rocks or on bedrock, when
bottom topography is complex or simply by the gear jumping off of the
bottom due to tidal currents. That is, sampling gear is biased.
Furthermore, it is not just sampling gear that is problematic. Survey
design is biased: in design, each areal unit is expected to be
homogeneous in space AND time. External factors being factored out such
as bottom temperatures, food availability and predation, aggregation
behavior are not homogeneous, no matter how well designed a sampling
design may be. Further, there is the notion of *trawlable bottom*: some
areas cannot be sampled without tearing or losing nets. These areas are
of course not sampled and only habitats that are easier to sample will
be used and consequently, over represented. Though allocation may be
*random* in design and theory, in application, it is at best, *almost
random* and depending upon bottom type, usually biased.

As such, the observation model is also more complex in *Model 2*. The
assumption of zero values in survey abundance indicating a true zero is
a very strong assumption. This is especially tenuous when it is known
that surveys sample many size ranges very poorly. The observation model
of *Model 2* is, therefore, a simple linear regression. Of course,
higher order terms and other covariates can and should enter into the
observation model to account for potential survey and behavior induced
bias. In this paper, this is mostly accomplished via the statistical
abundance index model (\cite{Choi_et_al_2022}). Some additional freedom is
given in the observation model in that the prior for the intercept term
$c_{i}$ was informed by $\mu_{i}$, the fraction of the minimum observed
value relative to the maximum value.

\begin{align*}
  m_i &\sim N(1.0, 0.1) \\
  c_i &\sim N(\mu_i, 0.1) \\
  y_{\mathit{ti}} &\sim N(\mu_i n_{\mathit{ti}} + c_i, \sigma_{ti})
\end{align*}

Here, ${y_{\mathit{ti}}={Y_{\mathit{ti}}/\mathit{\max}}}{(Y_{i})},$
constrains abundance to the interval (0, 1) and helps to confer better
numerical properties (faster and more stable optimization, integration)
in that variables are of similar magnitudes. Additionally, the
variability of the observations was propagated into *Model 2* by
assuming that the coefficient of variation of observations were modal
estimates of the observation model using a Lognormal (LN) prior with a
mode of log(CV) specific to each component $i$ and survey time $t$ and a
SD of 0.25:

$$\sigma_{ti} \sim \mathrm{LN}(\log(\mathrm{CV}_{ti}), 0.25).$$

This works as $y_{\mathit{ti}}$ are already normalized to (0,1). The
other priors used for the *Model 2* are also informative based upon
expected biological constraints and some very wide distributions for the
variance components to provide well mixed posterior samples as
determined by Effective Sample Size (given autocorrelation in samples
within chains) and *Rhat* (convergence criterion across chains):


\begin{align*}
  K &\sim\mathrm{LN}\left(\log(\kappa),0.25 \right) \\
  b &\sim\mathrm{LN}\left(\log(1),0.5 \right) \\
  d &\sim\mathrm{LN}\left(\log(0.22),0.25 \right) \\
  \psi &\sim\mathrm{LN}\left(\log(0.49),0.5 \right) \\
  v &\sim\mathrm{LN}\left(\log(1.46),0.5 \right)
\end{align*}

Fixed reference points analogous to the concepts of MSY and FMSY are not
readily identifiable in *Model 2* as "production" is now dependent upon
multiple system components each of which have externally imposed
effective habitat viability trends and related natural and fishing
mortalities (perturbations). That is, they are nonstationary, in the
dynamical sense. It is, however, possible to compute a related concept,
which we will call the *fisheries footprint*; it is computed as the
difference in trajectories between the system state estimated for the
fishable component with and without fishing. The biological parameters
are estimated under conditions of fishing activity. In the absence of
fishing, these parameters may be expected to be different, especially
when fishing activity is extreme. In some cases, maturity and growth
schedules can shift
(\cite{Zwanenburg_et_al_2002}; \cite{Choi_et_al_2004}; \cite{Choi_et_al_2005a}). The
assumption here is that extreme fishing activity has not occurred with
sufficient pressure nor time to generate phenotypic or genotypic change.
The difference in predicted trajectories between the fished and unfished
conditions, therefore, provides a crude, first order estimate of the
impact of fishing, without assumptions beyond that the model reasonably
approximates reality. This *fisheries footprint* is placed on a relative
scale from 0 to 1, by normalizing with the expected unfished abundance.
As such, the *fisheries footprint* identifies the fraction of potential
biomass that was reduced by fishery exploitation.

# Results and Discussions

Estimates of abundance (biomass) of the exploited component are shown
for each region for each model in
Fig. \ref{figpredictions}. In NENS, the discrete *Model 1* showed
two troughs in (pre-fishery) abundance in 2005 and 2017. The surveyed
index after adjustment for the observation model tends to be lower
relative to the pre-fishery abundance, which is consistent with the
removal of biomass during the fishing season. The exception to this
pattern was in late 2013 and 2019 when strong recruitment was also
observed. *Model 2* shows a similar pattern of fall fishable biomass,
though with a reduced magnitude. The post 2019 period showed significant
divergence relative to the survey index and especially in 2020 when no
surveys were completed due to Covid-19 related disruptions. SENS also
demonstrated a similar periodicity to NENS in both models
(Fig. \ref{figpredictions}). *Model 2* suggests a dampened time
series, attributable to the dynamics of the recruitment being estimated
to be flat (Fig. \ref{figm2f}). In CFA 4X, *Model 1* suggests an important
decline following a peak in 2010. *Model 2* suggests a similar
trajectory, though one that is much more variable. This area is subject
to crab movement from the adjoining area as well as spatial aggregation
due to extreme temperature conditions, elevated mortality from other
predators and disease. *Model 2* also suggests overall abundance that is
much lower than the very optimistic expectations of *Model 1*.

![Posterior median biomass (dark orange) for each area of study and
model. Posterior realizations of dynamics are shown to demonstrate
variability of solutions. For *Model 1* (left), *pre-fishery* posterior
estimates of fishable biomass are shown. For *Model 2*, the posterior
estimates of fishable biomass are shown in orange. Overlaid (gray) in
both are survey abundance estimates (*post-fishery*, except prior to
2004) after correction for their respective observation models. Overlaid
in green are posterior samples of abundance trajectories expected had
there been no fisheries exploitation (green); and dark green identifies
their respective means. \label{figpredictions}](media/predictions.png){#figpredictions
width=80%}

It is important to note that *Model 1* will project an increase in
abundance, even if recruitment does not exist. It will only project a
decline if abundance is above carrying capacity. This structural
expectation is of course unrealistic. *Model 2*, which accounts for
recruitment, navigates this problem a little more sensibly as an
increase can only occur if recruitment exceed mortality. For example, it
projects for CFA 4X, a decline, even in the absence of fishing. In SENS,
abundance is projected to increase slightly in the absence of fishing.
Only NENS was projected to have a rapid increase in abundance
(Fig. \ref{figpredictions}).

Another observation is that abundance may be overestimated by *Model 1*.
For example, in CFA 4X where we have supporting information of very high
natural mortality rates associated with predation and environmental
variability, through the loss of strong year classes, abundance
estimates are extremely optimistic. The same issues also exist for NENS,
where strong adolescent crab year classes seem to disappear at a rate
faster than might be expected, possibly due to predation. By extension,
SENS also likely exhibits overly optimistic time trends, given the very
strong year classes diminishing rapidly before entry into the fishable
component. As a result of this potential overestimation of abundance,
fishing mortality estimates by *Model 1* may in fact be overly low.
*Model 2*'s solutions seem more reasonable given the supporting
contextual information known of the different areas. The overall
relative shapes of abundance and fishing mortality are, however, similar
(Figs. \ref{figpredictions}, \ref{figmort}).

![Instantaneous fishing mortality estimated from *Model 1* and 2 across
time. The overall patterns of fishing mortality are similar across
models. However, the overall magnitudes are much higher in *Model 2*.
Fishing in CFA 4X was closed in 2018 due to low abundance estimates and
warm environmental conditions. \label{figmort}](media/fishing_mortality.png){#figmort
width="10.5 cm"}

The continuous form of *Model 2* also permits us to infer the subannual
dynamics of the snow crab. The saw tooth pattern of abundance
(Fig. \ref{figpredictions}) suggests that during the fishing season,
the rate of exploitation far surpasses the slower rate of growth of the
fishable component. It also identifies how important temporal aliasing
might be if observations of landings or survey sample times are not
properly accounted. The fishery footprint across the different areas
(Fig. \ref{figfoot}) indicates that they are very similar to fishing mortality in overall
form (as they should be, as the numerator, catch are the same). The
former is on a scale that is the same as abundance and so simpler to
understand than the exponential scale of an instantaneous rate. *Fishery
footprint* also has a reasonably direct relationship to a question that
is often asked in applied situations: *How much of an effect is fishing
activity having?* In the case of the areas studied, the *fishery
footprint* has declined to conservative and manageable levels (that is,
a small fraction of the total available abundance) since the mid-2000's.
SENS, in particular, has been consistently sustainable in the historical
record. In the presence of strong environmental variability, it will be
necessary to continue to maintain a small *fisheries footprint* to
reduce the susceptibility of the species such forces as has been the
case for Atlantic cod and other collapsed fisheries (\cite{Choi_2022}). This
is especially the case as heavy exploitation in the long-term can result
in reduced reproductive success of females; loss of dominance by
invasion of habitat by competitors and even genetic/phenotypic selection
for reduced size at maturity (\cite{Comeau_Conan_1992}; \cite{Choi_2011}).

![Posterior realizations of Fisheries footprint for each area of study
derived from *Model 2*. \label{figfoot}](media/footprint.png){#figfoot width=80% height=80%}

The first ever estimate of potential recruitment ($M_{2}$) and mature
female abundance ($F$) for the area in focus are shown in
Fig. (\ref{figm2f}). It shows that *Model 2* has difficulty tracking these components. However,
peaks and troughs are identified and they seem to be coherent across
areas. The lack of concordance with the observations suggest that other,
as yet unmodeled processes likely need to be better accounted. The most
likely missing processes include movement (inshore-offshore,
shallow-deep) as well as predation, as the areas of study are not *well
mixed* and show spatiotemporal structure and increases in the relative
abundance of predators such as Atlantic Halibut (\cite{DFO_2021}). Smaller
areal units would also better parameterize the spatiotemporal
heterogeneity observed in the latent biological processes known to occur
in the area (\cite{Choi_2011}).

![*Model 2* posterior numerical abundance for $M_{2}$ (recruits) and $F$
(mature female). Overlaid (gray) are survey abundance estimates
(post-fishery, except prior to 2004) after correction for observation
models. \label{figm2f}](media/m2_fem.png){#figm2f
width="13.5 cm"}

\clearpage

# Conclusions

In studies of exploited marine animal populations, most models focus
upon age structure (\cite{Quinn_2003}). Most also tend to be temporally
discrete (annual) difference equation approximations. This is largely
due to the annual cycle in temperate regions of exploitation, and data
assimilation, survey and assessment cycles and historically due to
limitations to computational capacity. With such discretizations, come
various approximations and assumptions and ultimately potential error or
bias from temporal (and spatial) aliasing. Space is treated as an
externality or more usually, completely ignored, though of course,
advection-diffusion differential equation models are readily formulated.

The continuous approach that we developed in this study addresses
mechanisms in a structured manner. The additional model complexity
succeeds in providing a similar and perhaps more reasonable
understanding of snow crab population dynamics than the classical
biomass dynamics model. It also permits a promising way forward towards
describing the *fishery footprint* measured relative to potential,
non-stationary abundance in the absence of fishing. It also identifies
the importance of accounting for temporal dynamics with minimal aliasing
in survey and landings, and the difficulties that may still need to be
addressed: most notably the *non-well-mixed* nature of the areas studied
show spatial and temporal structures that require further
parameterization and development.



---
 

## The main modelling tools used:
  
  * [Julia](https://julialang.org/) 
  * [Turing](https://turing.ml/stable/)
  * [SciML](https://docs.sciml.ai/Overview/stable/)
  * [DifferentialEquations](https://github.com/SciML/DifferentialEquations.jl)
  * Other libraries used can be found in the set up environment (*.environment.jl) for each project

---

## Data requirements: 

If data files have not already been created, then do so. Data files are generated by :

  - [snowcrab_size_structured_dde_biomass_index_carstm.md](https://bitbucket.com/autocatalysis/model_fishery/scripts/snowcrab_size_structured_dde_biomass_index_carstm.md)  
  

## Define key directories and main run level options: 

These directories need to be defined. 

```{julia}
#| eval: false 
#| output: false

using AssessmentTools
using AssessmentTools.DelayDifferential

# Model methods: default_snowcrab_dde_config, simulate_dde_system,
# bayesian_delay_differential_model, simulate_delay_difference,
# plot_state_variables, plot_fishery_predictions, plot_fishery_footprint

# run-time parameters 
year_assessment = 2024   # <<<<<<<<-- change
yrs = 2000:year_assessment  


model_variation = "size_structured_dde_normalized" 

  #= currently implemented models:
    model_variation = "size_structured_dde_normalized"    # 24hrs, >24 hrs
    model_variation = "size_structured_dde_unnormalized"  # basic continuous model without normalization ... very very slow .. do not use (incomplete params need tweaking) ::   24hrs, >24 hrs .. needs to be updated
  =#

aulab ="choose one from below"    

# area to model .. choose a region of interest:
  #= currently implemented regions:
    aulab ="cfanorth"    
    aulab ="cfasouth"    
    aulab ="cfa4x"       
  =#


# input/output locations

# outputs_dir = projectdir( homedir(), "bio.data", "bio.snowcrab",  "fishery_model", "outputs", string(year_assessment), model_variation )
outputs_dir = projectdir( "outputs", string(year_assessment), model_variation )
  mkpath(outputs_dir)
  print( "outputs_dir: ", outputs_dir, "\n\n" )

# mcmc save file name and location
res_fn = joinpath( outputs_dir, string("results_turing", "_", aulab, ".hdf5" ) )  
print( "results file:",  res_fn )


# model input data, remember to save/copy data to this location (manually or in R)
fishery_data = load( projectdir( "data", "biodyn_number_size_struct.RData"  ), convert=true)

 

```

 
## Sample from posteriors after defining sampler  

Determine Turing run parameters that work reasonably and obtain posterior samples of parameter estimates.
 

```{julia}
#| eval: false 
#| output: false

# include( srcdir( "size_structured_dde_functions.jl" ) ) 
 

# set up data and parameters and put in Global namespace
PM, M, solver_params  = size_structured_dde_data( yrs, aulab, fishery_data, model_variation )
 
Random.seed!(year_assessment);

# Turing.setprogress!(false);

rand(M) # above instantiates model as M: make sure M works
      
# .. make sure to check autocorrelation where required and thin 

# adjust parameters for specific model_variation and get model setup
 
# prior sampling seems broken 2024-10-24 JC 
  #=
    prior = sample( M, Prior(), n_samples)
    init_params, vns = init_params_extract( prior ) 
    res = sample( M, Turing.SMC(), 100; init_params=init_params) # cannot be larger 
  =#

res = sample( M, Turing.SMC(), 1000 )

using Diffractor
res = sample( M, Turing.NUTS(; adtype=AutoDiffractor()), 100 ) 

# using Zygote
# res = sample( M, Turing.NUTS(; adtype=AutoZygote()), 100 )

using Enzyme
res = sample( M, Turing.NUTS(; adtype=AutoEnzyme()), 100 )
init_params, vns = init_params_extract( res ) 

# production: 
# Turing NUTS-specific default options  ..  see write up here: https://turing.ml/dev/docs/using-turing/sampler-viz
n_adapts, n_samples, n_chains = 3000, 5000, 4
target_acceptance_rate, max_depth, init_ϵ = 0.65, 7, 0.01
turing_sampler = Turing.NUTS( target_acceptance_rate; max_depth=max_depth, init_ϵ=init_ϵ )

# res = sample( M, turing_sampler, n_samples; init_params=init_params) # cannot be larger 
res = sample( M, turing_sampler, MCMCThreads(), n_samples, n_chains ; init_ϵ=init_ϵ, drop_warmup=true, progress=true, init_params=init_params ) # sample in parallel
 
# make sure to check autocorrelation where required and thin

# save results to (model_outdir) as a hdf5  .. can also read back in R as:  h5read( res_fn, "res" )

res_fn = joinpath( model_outdir, string("results_turing", "_", aulab, ".hdf5" ) )   # already defined in *.environment.jl .. here as a reminder
@save res_fn res

  #=  to reload a save file:

    @load res_fn res
    
  =#
  
# generate predicted trajectories and params of interest from posterior samples 
# n scaled, n unscaled, biomass of fb with and without fishing, model_traces, model_times 
m, num, bio, trace, trace_bio, trace_time = fishery_model_predictions(res ) 
Fkt, FR, FM = fishery_model_mortality()   # fishing (kt), relative Fishing mortality, instantaneous fishing mortality:

# quick check of fits and solutions:
# fishery_model_plot( toplot=("survey", "fishing"), alphav=0.025 )


# save a few data files as semicolon-delimited CSV's for use outside Julia
summary_fn = joinpath( model_outdir, string("results_turing", "_", aulab, "_summary", ".csv" ) )  
CSV.write( summary_fn,  summarize( res ), delim=";" )  # use semicolon as , also used in parm names

bio_fn1 = joinpath( model_outdir, string("results_turing", "_", aulab, "_bio_fishing", ".csv" ) )  
CSV.write( bio_fn1,  DataFrame(bio[:,:,1], :auto), delim=";" )  # use semicolon as , also used in parm names

bio_fn2 = joinpath( model_outdir, string("results_turing", "_", aulab, "_bio_nofishing", ".csv" ) )  
CSV.write( bio_fn2,  DataFrame( bio[:,:,2], :auto), delim=";" )  # use semicolon as , also used in parm names

fm_fn = joinpath( model_outdir, string("results_turing", "_", aulab, "_fm", ".csv" ) )  
CSV.write( fm_fn,  DataFrame( FM, :auto), delim=";" )  # use semicolon as , also used in parm names
 

## Figures

outformat = "png"  # svg, pdf, etc

#  pl = plot(pl, ylim=(0, 2.5))

pl = fishery_model_plot( toplot=("survey", "fishing", "nofishing", "trace"), alphav=0.025 )
# pl = plot(pl, ylim=(0, 1.5))
savefig(pl, joinpath( model_outdir, string("plot_predictions_everything_", aulab,  ,".", outformat ) )  )


pl = fishery_model_plot( toplot=("trace", "survey"), alphav=0.025 )
# pl = plot(pl, ylim=(0, 5.5))
savefig(pl, joinpath( model_outdir, string("plot_predictions_trace_", aulab,  ,".", outformat ) )  )

# annual snapshots of biomass (kt) 
pl = fishery_model_plot( toplot=("survey", "fishing" ) )
savefig(pl, joinpath( model_outdir, string("plot_predictions_", aulab,  ,".", outformat ) )  )

# annual snapshots of biomass (kt) 
pl = fishery_model_plot( toplot=("survey", "fishing", "nofishing") )
savefig(pl, joinpath( model_outdir, string("plot_predictions_full_", aulab,  ,".", outformat ) )  )

# plot fishing mortality
pl = fishery_model_plot( toplot="fishing_mortality" )
# pl = plot!(pl; ylim=(0, 1.6 ) )
savefig(pl, joinpath( model_outdir, string("plot_fishing_mortality_", aulab,  ,".", outformat ) )  )

# HCR plot
pl = fishery_model_plot( toplot="harvest_control_rule", n_sample=1000  )  # hcr
#  pl = plot!(pl; ylim=(0, 1.75 ), xlim=(0,6.5) )
savefig(pl, joinpath( model_outdir, string("plot_hcr_", aulab,  ,".", outformat ) )  )

# HCR footprint
pl = fishery_model_plot( toplot="harvest_control_rule_footprint", n_sample=1000  )  # hcr with fishing footprint
pl = plot!(pl; ylim=(0, 1.0 ) )
savefig(pl, joinpath( model_outdir, string("plot_hcr_footprint_", aulab,  ,".", outformat ) )  )
    
# fishery footprint
pl = fishery_model_plot( toplot="footprint" )
pl = plot!(pl; ylim=(0, 1.0 ) )
savefig(pl, joinpath( model_outdir, string("plot_footprint_", aulab,  ,".", outformat ) )  )

pl = fishery_model_plot( toplot="trace_footprint", alphav=0.02 )
pl = plot!(pl; ylim=(0, 1.0 ) )
savefig(pl, joinpath( model_outdir, string("plot_footprint_trace_", aulab,  ,".", outformat ) )  )

pl = fishery_model_plot( toplot="fishing_mortality_vs_footprint" )
pl = plot!(pl; ylim=(0, 1.0 ) )
pl = plot!(pl; xlim=(0, 2.0 ) )
savefig(pl, joinpath( model_outdir, string("plot_fishing_mortality_vs_footprint_", aulab,  ,".", outformat ) )  )

# timeseries of predictions (number; kn and pl =plot) -- not relevent if only 1 state varable
statevar = 1  # index of S
pl = fishery_model_plot( toplot="number", si=statevar )  # s1 as numbers
savefig(pl, joinpath( model_outdir, string("plot_predictions_timeseries_",  aulab, "_", statevar,  ,".", outformat ) )  )

statevar = 2  # index of S
pl = fishery_model_plot( toplot="number", si=statevar )  # s1 as numbers
savefig(pl, joinpath( model_outdir, string("plot_predictions_timeseries_",  aulab, "_", statevar,  ,".", outformat ) )  )

statevar = 3  # index of S
pl = fishery_model_plot( toplot="number", si=statevar )  # s1 as numbers
savefig(pl, joinpath( model_outdir, string("plot_predictions_timeseries_",  aulab, "_", statevar,  ,".", outformat ) )  )

statevar = 4  # index of S
pl = fishery_model_plot( toplot="number", si=statevar )  # s1 as numbers
savefig(pl, joinpath( model_outdir, string("plot_predictions_timeseries_",  aulab, "_", statevar,  ,".", outformat ) )  )

statevar = 5  # index of S
pl = fishery_model_plot( toplot="number", si=statevar )  # s1 as numbers
savefig(pl, joinpath( model_outdir, string("plot_predictions_timeseries_",  aulab, "_", statevar,  ,".", outformat ) )  )

statevar = 6  # index of S
pl = fishery_model_plot( toplot="number", si=statevar )  # s1 as numbers
savefig(pl, joinpath( model_outdir, string("plot_predictions_timeseries_",  aulab, "_", statevar,  ,".", outformat ) )  )


prior = sample( M, Prior(), 10000)
posterior = res

# bluegrey is prior, purple is posterior 
vn = "K[1]"
pl = plot_prior_posterior( vn, prior, posterior, bw=0.04)
save(joinpath( model_outdir, string("plot_prior_", vn, "_", aulab,  ,".", outformat ) ), pl  )

vn = "q1[1]"
pl = plot_prior_posterior( vn, prior, posterior, bw=0.04)
save(joinpath( model_outdir, string("plot_prior_", vn, "_", aulab,  ,".", outformat ) ), pl  )

vn = "q0[1]"
pl = plot_prior_posterior( vn, prior, posterior, bw=0.04)
save(joinpath( model_outdir, string("plot_prior_", vn, "_", aulab,  ,".", outformat ) ), pl  )

vn = "model_sd[1,1]"
pl = plot_prior_posterior( vn, prior, posterior, bw=0.04)
save(joinpath( model_outdir, string("plot_prior_", vn, "_", aulab,  ,".", outformat ) ), pl  )

vn = "b5[2]"
pl = plot_prior_posterior( vn, prior, posterior, bw=0.04)
save(joinpath( model_outdir, string("plot_prior_", vn, "_", aulab,  ,".", outformat ) ), pl  )

vn = "b6[2]"
pl = plot_prior_posterior( vn, prior, posterior, bw=0.04)
save(joinpath( model_outdir, string("plot_prior_", vn, "_", aulab,  ,".", outformat ) ), pl  )

vn = "d[1]"
pl = plot_prior_posterior( vn, prior, posterior, bw=0.04)
save(joinpath( model_outdir, string("plot_prior_", vn, "_", aulab,  ,".", outformat ) ), pl  )

vn = "d2[1]"
pl = plot_prior_posterior( vn, prior, posterior, bw=0.04)
save(joinpath( model_outdir, string("plot_prior_", vn, "_", aulab,  ,".", outformat ) ), pl  )

vn = "v[1]"
pl = plot_prior_posterior( vn, prior, posterior, bw=0.04)
save(joinpath( model_outdir, string("plot_prior_", vn, "_", aulab,  ,".", outformat ) ), pl  )

vn = "u0[1]"
pl = plot_prior_posterior( vn, prior, posterior, bw=0.04)
save(joinpath( model_outdir, string("plot_prior_", vn, "_", aulab,  ,".", outformat ) ), pl  )


# projections relative to status quo .. update the values below 
status_quo_tac = [0.97875, 3.9729 + 3.37212, 0.055 ][ki] # 2022
for frac in (0.8, 0.9, 1.0, 1.1, 1.2)
  
  print( "\nFraction: ", frac )

  Catch = status_quo_tac * frac 
  m, num, bio, trace, trace_bio, trace_time = project_with_constant_catch( 
    res, 
    solver_params=solver_params, 
    PM=PM, 
    Catch=Catch, 
    ny_fishing_pattern=5  
  )
  
  Fkt, FR, FM = fishery_model_mortality() 
  
  pl = fishery_model_plot( 
    toplot=("trace_projections"), alphav=0.025, time_range_predictions=(2002,year_assessment+4)  
  )

  savefig(pl, joinpath( model_outdir, string("plot_trace_projections_", aulab, "__", frac, "__", ,".", outformat ) )  )
  
  pl = fishery_model_plot( 
    toplot=("trace_footprint_projections"), alphav=0.025,  time_range_predictions=(2002,year_assessment+4)  
  )
  
  savefig(pl, joinpath( model_outdir, string("plot_trace_footprint_projections_", aulab, "__", frac, "__", ,".", outformat ) )  )

end


```

# End
---


~
~
~


---
### Tests of other methods of parameter estimation and notes (IGNORE)

```{julia}
#| eval: false 
#| output: false

restart_method = false
if restart_method
  using Optim, StatsBase
  res_map = Optim.optimize( M, MAP())  # find starting point (modes from maximum aposteriori)
  coeftable( res_map)
  res = sample( M, Turing.NUTS(), 30, init_params=res_map.values.array )
end


do_variational_inference = false
if do_variational_inference

  # to do Variational Inference (an sd term goes less than 0 .. not sure of the cause ):
  
  using DistributionsAD, AdvancedVI

  res_vi =  vi(M, Turing.ADVI( 10, 1000));

     # Run turing_sampler, collect results. @doc(Variational.ADVI) :
     # samples_per_step::Int64
     # Number of samples used to estimate the ELBO in each optimization step.
     # max_iters::Int64
     # Maximum number of gradient steps.

  res_vi_samples = rand( res_vi, 1000)  # sample via simulation

  p1 = histogram(res_vi_samples[1, :]; bins=100, normed=true, alpha=0.2, color=:blue, label="")
  density!(res_vi_samples[1, :]; label="s (ADVI)", color=:blue, linewidth=2)
  density!(res, :s; label="s (NUTS)", color=:green, linewidth=2)
  vline!([var(x)]; label="s (data)", color=:black)
  vline!([mean(res_vi_samples[1, :])]; color=:blue, label="")

  p2 = histogram(res_vi_samples[2, :]; bins=100, normed=true, alpha=0.2, color=:blue, label="")
  density!(res_vi_samples[2, :]; label="m (ADVI)", color=:blue, linewidth=2)
  density!(res, :m; label="m (NUTS)", color=:green, linewidth=2)
  vline!([mean(x)]; color=:black, label="m (data)")
  vline!([mean(res_vi_samples[2, :])]; color=:blue, label="")

  plot(p1, p2; layout=(2, 1), size=(900, 500))


  do_maximum_likelihood = false
  if do_maximum_likelihood
    res_mle = Turing.optimize(M, MLE())
    res_mle = Turing.optimize(M, MLE())
    res_mle = Turing.optimize(M, MLE(), NelderMead())
    res_mle = Turing.optimize(M, MLE(), SimulatedAnnealing())
    res_mle = Turing.optimize(M, MLE(), ParticleSwarm())
    res_mle = Turing.optimize(M, MLE(), Newton())
    res_mle = Turing.optimize(M, MLE(), AcceleratedGradientDescent())
    res_mle = Turing.optimize(M, MLE(), Newton(), Optim.Options(iterations=10_000, allow_f_increases=true))

    using StatsBase
    coeftable(res_mle)
end


do_maximum_aposteriori = false
if do_maximum_aposteriori
  res_map = Turing.optimize(M, MAP())
  using StatsBase
  coeftable(res_map)
end


 
rng = MersenneTwister(26)
resp = predict(rng, textmodel_marginal_pred(data), res)
@df resp ecdfplot(:"b5[1]"; label="birth rate 1")



using Flux, DiffEqFlux
params = Flux.params(p)

msol =  solve( prob,  solver, callback=cb, saveat=dt)    # isoutofdomain=(y,p,t)->any(x->x<0,y)  # to force positive

function predict_rd() # Our 1-layer "neural network"
  solve(prob, solver, p=p, saveat=dt)[1] # override with new parameters  # isoutofdomain=(y,p,t)->any(x->x<0,y)  # to force positive
end

loss_rd() = sum(abs2,x-1 for x in predict_rd()) # loss function (squared absolute) x-1

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cbflux = function () #callback
  # function to observe training
  display(loss_rd())
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,p=p), solver,saveat=dt), ylim=(0,kmu*2)))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_rd, params, data, opt, cb = cbflux)

m = Chain(
  Conv((2,2), 1=>16, relu),
  x -> maxpool(x, (2,2)),
  Conv((2,2), 16=>8, relu),
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)),
  x -> solve(prob, solver,u0=x,saveat=0.1)[1,:],
  Dense(288, 10), softmax) |> gpu

```




# A quick view of results generated above

--- these need to be created and made ....


And that is it. Now we continue to look at some of the main results. This part gets processed as a Quarto document. See Premable (above for instructions on creating a report). 



```{r}
#| eval: true
#| output: false
#| echo: false
#| label: setup

require(knitr)

knitr::opts_chunk$set(
  root.dir = data_root,
  echo = FALSE,
  out.width="6.2in", 
  fig.retina = 2,
  dpi=192
)
 

require(ggplot2) 

require(aegis)  # basic helper tools
 
year.assessment = params$year_assessment 

year_previous = year.assessment - 1

p = bio.snowcrab::load.environment( year.assessment=year.assessment )  

rootdir = file.path("/home", "jae" )
 
require(gt)  # table formatting

outtabledir = file.path( p$annual.results, "tables" )

years = as.character(1996: year.assessment)

lregions = list(region=c("cfanorth", "cfasouth", "cfa4x"))
reg_labels = c("N-ENS", "S-ENS", "CFA 4X")  # formatted for label
 
regions = unlist(lregions)
nregions = length(regions)
 
# in case of local changes
loadfunctions( "aegis")
loadfunctions( "bio.snowcrab")  # in case of local edits
 
p$corners = data.frame(plon=c(220, 990), plat=c(4750, 5270) )
p$mapyears = year.assessment + c(-5:0 )   # default in case not specified
 

loc = file.path( rootdir, "projects", "model_fishery", "outputs", year.assessment, params$model_variation )
 
```

  



## Stock status: Biomass Index (aggregate) {.c}
    
```{r}
#| results: asis
#| echo: false
#| eval: true
#| output: true
#| fig-dpi: 144
#| fig-height: 4
#| label: fbindex-timeseries

fn = file.path( rootdir, "bio.data", "bio.snowcrab", 'modelled', 'default_fb', 'aggregated_biomass_timeseries' , 'biomass_M0.png')

include_graphics( fn )

```

The fishable biomass index (t) predicted by CARSTM of Snow Crab survey densities. Error bars represent Bayesian 95\\% Credible Intervals. Note large errors in 2020 when there was no survey.
 

## Stock status: Modelled Biomass (pre-fishery) {.c}
```{r}
#| results: asis
#| echo: false
#| eval: true
#| output: true
#| fig-dpi: 144
#| fig-height: 4 
#| label: logisticPredictions

 
for (r in 1:nregions){
  reg = regions[r]
  REG = reg_labels[r] 
  cat("#### ", REG, "\n")
  fn = file.path( loc, paste("plot_predictions_", reg, ".png", sep="" ) ) 
  show_image(fn) 
  cat("\n\n")
} 


```
Model 1 fishable, posterior mean modelled biomass (pre-fishery; kt) are shown in dark orange. Light orange are posterior samples of modelled biomass (pre-fishery; kt) to illustrate the variability of the predictions. The biomass index (post-fishery, except prior to 2004) after model adjustment by the model catchability coefficient is in gray.



## Stock status: Fishing Mortality ... {.c}

```{r}
#| results: asis
#| echo: false
#| eval: true
#| output: true
#| fig-dpi: 144
#| fig-height: 4
#| label: logisticFishingMortality



for (r in 1:nregions){
  reg = regions[r]
  REG = reg_labels[r] 
  cat("#### ", REG, "\n")
  fn = file.path( loc, paste("plot_fishing_mortality_", reg, ".png", sep="" ) ) 
  show_image(check_file_exists(fn)) 
  cat("\n\n")
} 

``` 
Time-series of modelled instantaneous fishing mortality from Model 1. Samples of the posterior densities are presented, with the darkest line being the mean.


## Stock status: Reference Points {.c}

```{r}
#| results: asis
#| echo: false
#| eval: true
#| output: true
#| fig-dpi: 144
#| fig-height: 4
#| label: logistic-hcr
 
for (r in 1:nregions){
  reg = regions[r]
  REG = reg_labels[r] 
  cat("#### ", REG, "\n")
  fn = file.path( loc, paste("plot_hcr_", reg, ".png", sep="" ) ) 
  show_image(check_file_exists(fn)) 
  cat("\n\n")
} 
  

``` 

Reference Points (fishing mortality and modelled biomass) from Model 1. The large yellow dot indicates most recent year and stars the 95\\% CI.



## Surplus production ("Shaeffer form") {.c}

```{r}
#| results: asis
#| echo: false
#| eval: true
#| output: true
#| fig-dpi: 144
#| fig-height: 4
#| label: logistic-surplus-production

for (r in 1:nregions){
  reg = regions[r]
  REG = reg_labels[r] 
  cat("#### ", REG, "\n")
  fn = file.path( loc, paste("plot_surplus_production_", reg, ".png", sep="" ) ) 
  show_image(check_file_exists(fn)) 
  cat("\n\n")
} 
  
``` 

Surplus production from Model 1. Each year is represented by a different colour.



 
## Prior-posterior comparsons: Carrying capacity (K; kt) {.c}

```{r}
#| results: asis
#| echo: false
#| eval: true
#| output: true
#| fig-dpi: 144
#| fig-height: 4
#| label: logistic-prior-posterior-K

for (r in 1:nregions){
  reg = regions[r]
  REG = reg_labels[r] 
  cat("#### ", REG, "\n")
  fn = file.path( loc, paste("plot_prior_K_", reg, ".png", sep="" ) ) 
  show_image(check_file_exists(fn)) 
  cat("\n\n")
} 
  
``` 

Prior-posterior comparisons from Model 1. Carrying capacity (K; kt).


## Prior-posterior comparsons: Intrinsic rate of biomass increase (r) {.c}

```{r}
#| results: asis
#| echo: false
#| eval: true
#| output: true
#| fig-dpi: 144
#| fig-height: 4
#| label: logistic-prior-posterior-r
 
for (r in 1:nregions){
  reg = regions[r]
  REG = reg_labels[r] 
  cat("#### ", REG, "\n")
  fn = file.path( loc, paste("plot_prior_r_", reg, ".png", sep="" ) ) 
  show_image(check_file_exists(fn)) 
  cat("\n\n")
} 

``` 

Prior-posterior comparisons from Model 1. Intrinsic rate of biomass increase (r).


 
## Prior-posterior comparsons: Catchability coefficient (q) {.c}

```{r}
#| results: asis
#| echo: false
#| eval: true
#| output: true
#| fig-dpi: 144
#| fig-height: 4
#| label: logistic-prior-posterior-q
 
for (r in 1:nregions){
  reg = regions[r]
  REG = reg_labels[r] 
  cat("#### ", REG, "\n")
  fn = file.path( loc, paste("plot_prior_q1_", reg, ".png", sep="" ) ) 
  show_image(check_file_exists(fn)) 
  cat("\n\n")
} 
   
``` 

Prior-posterior comparisons from Model 1. Catchability coefficient (q).



 
## Prior-posterior comparsons: Observation error  {.c}

```{r}
#| results: asis
#| echo: false
#| eval: true
#| output: true
#| fig-dpi: 144
#| fig-height: 4
#| label: logistic-prior-posterior-obserror

for (r in 1:nregions){
  reg = regions[r]
  REG = reg_labels[r] 
  cat("#### ", REG, "\n")
  fn = file.path( loc, paste("plot_prior_bosd_", reg, ".png", sep="" ) ) 
  show_image(check_file_exists(fn)) 
  cat("\n\n")
} 

``` 

Prior-posterior comparisons from Model 1. Observation error (kt).


 
## Prior-posterior comparsons: Model process error {.c}

```{r}
#| results: asis
#| echo: false
#| eval: true
#| output: true
#| fig-dpi: 144
#| fig-height: 4
#| label: logistic-prior-posterior-processerror

for (r in 1:nregions){
  reg = regions[r]
  REG = reg_labels[r] 
  cat("#### ", REG, "\n")
  fn = file.path( loc, paste("plot_prior_bpsd_", reg, ".png", sep="" ) ) 
  show_image(check_file_exists(fn)) 
  cat("\n\n")
} 
  
``` 

Prior-posterior comparisons from Model 1. Model process error.


# End 

  

      
 
