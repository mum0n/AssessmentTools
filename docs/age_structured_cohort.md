---
title: "Age-structured cohort-based population models"
author: "Jae S. Choi"
date: last-modified
date-format: "YYYY-MM-D"
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
params:
  year_assessment: 2024
  model_variation: Maximum_likelhood_basic
---
 
   
<!-- Preamble


This is a Markdown document ... To create HTML or PDF, etc, run: 

As it runs R and Julia, Quarto is probably the better tool to render. Alternatively, you can run everything directly and then render just the figures by turning off the R and Julia code chunks. To create an HTML via Quarto:

>> cd ~/projects/model_fishery/docs

>> make quarto FN=age_structured_cohort YR=2024 SOURCE=~/projects/model_fishery/scripts WK=~/projects/model_fishery/outputs DOCEXTENSION=html PARAMS="-P year_assessment:2024 -P model_variation:Maximum_likelhood_basic"

>> make quarto FN=age_structured_cohort YR=2024 SOURCE=~/projects/model_fishery/scripts WK=~/projects/model_fishery/outputs DOCEXTENSION=html PARAMS="-P year_assessment:2024 -P model_variation:Maximum_aposteriori"
 
  
Alter year and directories to reflect setup or copy Makefile and alter defaults to your needs.
   
-->

---
 
# Age-structured cohort-based population  


Age-structured models that track cohorts (age-groups) over time is/was used commonly in fisheries applications to estimate population structure/demographics from fisheries catch data as that is the simplest data to gather. 

Cohort Analysis is the core concept of most age-structured methods (attributed to Pope): 

- make assumptions on mortality (fishing and natural), usually of older animals (better captured) and plus groups 

  - Natural mortality assumed at 0.2.
  - Fishing_mortality at age is assumed constant for 7 and 8+.

 - Work backwards using numbers of these groups 
 - Exceptions: Sequential Pop. Anal., Stock synthesis (forward methods).


Generally, catch is assumed to be known without error. Cohort populations were estimated via back-calculation/propagation/projection ("Cohort Analysis") with simple assumptions (catch + cohort) to help re-create a "virtual" population (Fry 1949, based on Derzhavin 1922):

$N_{a+1,y+1} = N_{a,y} - C_{a,y} - Z_{a,y}$,

for a given age $a$ and year $y$, in discrete form. 

**For each cohort**, refining of the above to instantaneous total mortality ($Z$) as a combination of natural ($M$) and fishing ($F$) mortality and catachability ($q$) is attributed to Ricker, Beverton, Holt, and Paloheimo. Gulland (1965) is attributed this "backwards" mode which provided more convergent solutions (stable); Murphy (1965) estimated fishing mortality at early ages in forward mode which was found to give more divergent solutions (unstable). As such backwards mode was generally more favoured.

Beginning with the Baranov catch equation:

$C_t = [F_t / (F_t + M) ] \; N_t (1 - e^{-Z_t} )$

and a first order, nonlinear, decay equation:

$N_{t+1} = N_t \; e^{-Z_t} \; \; <=> \; \; N_t = N_{t+1} \; e^{Z_t}$,

gives in terms of catch, "terminal" fishing mortality and natural mortality:

$N_t = C_t / [ ( F_t / Z_t ) (1 - e^{-Z_t}) ]$ 

at $y=t$ the terminal year and Terminal fishing mortality $F_t$ and total mortality $Z_t=F_t + M_{(t)}$, where M is usually a constant but can be variable with time or age.

A further simplification, to avoid nonlinear regression, is the use of a step function for fishing mortality that is assumed to occur in one instant at exactly 1/2 through a time cycle (usually 1 year) and background natural mortality at a constant rate; this approximation is attributed to Pope (1972):

$N_{t+0.5} = N_t \; e^{-M/2} - C_t  \;$

$N_{t+1} = N_{t+0.5} \; e^{-M/2}$

$\quad \quad \; = [N_t \; e^{-M/2} - C_t ] \; e^{-M/2}$

which essentially amounts to [natural mortality before fishing event - catch] * natural mortality of remainder. Generally results are more stable with Z < 1 and M < F. It simplifies to:

$N_{t+1}  = N_t \; e^{-M} - C_t \; e^{-M/2}$

Solving for $N_t$ gives:

$N_{t} = N_{t+1} \; e^M + C_t \; e^{M/2}$, 

and fishing mortality can be estimated as:

$F_t = \log( | N_t / N/{t+1}| ) - M$

Various methods of estimating M exist. A useful one is to express Z vs effort. The Y-intercept provides the magnitude of background natural mortality and the slope represents the catchability coefficient q.

Additional structure in the form of distributional constraints that were "tuned" to an independent index of abundance, were known as ADAPT-VPA's and with multiple indices, as Integrated Catch Analysis. With the use of priors (e.g., Bayesian modelling) continued evolution/refinement of numerical methods and assumptions on age and maturity could be made, see eg, Millar and Meyer (2000) CJFAS 57:43-50 (Sequential Population Analysis).

All these methods assume age structure is reasonably well represented and errors in ageing of animals and usually a fixed natural mortality assumption and that catch is observed without error, no movement and no discards. They are important sources of uncertainty.


## Good resources: 

https://people.uncw.edu/scharff/courses/458/Lecture%2013%20-%20cohort%20models.pdf


 
---

# Age-based fishery modelling tools 

The main modelling tools used here:
  
  * [Julia](https://julialang.org/) 
  * [Turing](https://turing.ml/stable/) 
  * Other important libraries are found in the set up "environment" files



## Define key directories and main run level options and create input data: 

To begin, define starting directories and the example data (source: FAO tutorial's VPA data for Baltic Cod: Div 25-32): 


```{julia}
#| eval: false 
#| output: false

using AssessmentTools
using AssessmentTools.CohortAnalysis

# Model methods: popes_cohort_analysis, standard_vpa, adapt_vpa,
# bayesian_cohort_analysis, bayesian_adapt_vpa, and plot_state_variables



ages, yrs, C, A_yrs, A, body_weight_kg, maturity_ogive = data_baltic_cod_fao()

yi = findall(x-> x in A_yrs, yrs)  # overlapping years

Cobs = Float64.(C) # used where Integer is not appropriate

  #=
    C = catch, numbers at age x year
    A = abundance index, numbers age x year
    the remaining variables should be self-explanatory 
  =#


```
 
## Cohort Analysis

Cohort Analysis is the core concept of most age-structured methods (attributed to Pope): 
- make assumptions on mortality (fishing and natural), usually of older animals (better captured) and plus groups 
  - Natural mortality assumed at 0.2.
  - Fishing_mortality at age is assumed constant for 7 and 8+.
 - Work backwards using numbers of these groups 
 - Exceptions: Sequential Pop. Anal., Stock synthesis (forward methods).
 

```{julia}
#| eval: false 
#| output: false


 
# dimensions and initial conditions
na, ny = size(C)
iplus = [na-1, na]  # indices of plus size/age

# guess of natural mortality
M = 0.2   
# M = repeat([0.2], na)   # for ICA, VPA

# create F based on following conditions
Fp = repeat([0.5], ny)  # fishing mortality across yr for the plus groups 
Ft = [ 0.1, 0.2, 0.4, 0.5, 0.5, 0.5, 0.5 ] # Fishing mortality across age for the terminal year
F = setup_F( Fp, Ft, type="nonseparable", iplus=iplus )

# estimate N from F and M 
N0 = setup_N( C, F, M; iplus=iplus )

# deterministically (without error processes) back-solve with above starting conditions
N = cohort_analysis( N0, C, M; iplus=iplus  ) 

F = compute_fishing_mortality!(F, N, M; iplus=iplus)  # update F with finalized N's
Fall, fsubset, pl = summary_fishing_mortality(F, subset=1:5)
pl

B = N .* body_weight_kg[:,1]
Bmat, Btot, Bssb, pl = compute_biomass( B, maturity_ogive )
pl

Cpred = catch_cohort_estimate( N, F, M ) 

  
```

## Virtual Population Analysis (VPA)

VPA's tune Cohort Analysis results by assuming separability of fishing mortality by age and year and estimating these parameters.

Tuning is/was accomplished by least squares or Maximum-likelihood methods (respecting distributional forms of data and parameters). In the basic VPA, fishery independent estimates (A) are tuned to abundance estimates from Cohort Analysis by fitting a catchability coefficient (q) and Fishing mortality estimates of the most recent year (Fp) and oldest groups (Ft).
 
Tuning to surveys and other sources of information are called ADAPT and Integrated Catch Analysis. 

Tuning in the reverse direction is called Sequential Population Analysis.

So, approach is to wrap Cohort Analysis with random initial F estimates:

```{julia}
#| eval: false 
#| output: false


Fp0 = repeat([0.5], ny)
Ft0 = [ 0.1, 0.2, 0.4, 0.5, 0.5, 0.5, 0.5 ] 
q0 = repeat([1.0], na )

Random.seed!(1);

  #=
    test = rand(arraydist( LogNormal.( log.(Fp0), 0.25) ))
    test = rand(arraydist( LogNormal.( log.(Ft0), 0.25) ))

    histogram(test)
  =#


Fishing_mortality_assumption = "nonseparable"
Fishing_mortality_assumption = "separable"


# implemented models .. add more using the same template
#=
  fmod = Virtual_Population_Analysis_basic( Fishing_mortality_assumption, M, Cobs, iplus )  # not exactly a classical VPA as Ft and Fp are random effects

  fmod = Virtual_Population_Analysis( Fishing_mortality_assumption, M, Cobs, iplus )  # not exactly a classical VPA as Ft and Fp are random effects

  fmod = Adaptive_Virtual_Population_Analysis( Fishing_mortality_assumption, M, C , iplus)
  
  Ma = repeat([0.2], na)
  fmod = Integrated_Catch_Analysis( Fishing_mortality_assumption, Ma, C , iplus )
  
  fmod = Sequential_Population_Analysis( Fishing_mortality_assumption, M, C, iplus  ) # untested and needs assimilation.
=#

 

 
Optim.Options(iterations=1000)  # change options as required  

res = optimize( fmod, MLE(), NelderMead() )    # alt engines: BFGS(), Newton(), etc .. and MLE(), MAP()


Fp = [ res.values[Symbol("Fp[$i]")] for i in 1:ny ]
Ft_hat = [ res.values[Symbol("Ft[$i]")] for i in 1:na ]
F = setup_F( Fp, Ft_hat, type=Fishing_mortality_assumption, iplus=iplus )
N = setup_N( C, F, M; iplus=iplus )
N = cohort_analysis( N, C, M; iplus=iplus  ) 

F = compute_fishing_mortality!(F, N, M; iplus=iplus)  # update F with finalized N's
Fall, fsubset, pl = summary_fishing_mortality(F, subset=1:5)
pl
B = N .* body_weight_kg[:,1]
Bmat, Btot, Bssb, pl = compute_biomass( B, maturity_ogive )
pl
Cpred = catch_cohort_estimate( N, F, M ) # Note some negative predicted catches 



#= 
  # to get sampled estimates of N (if they exist)
  
  q = [ res.values[Symbol("q[$i]")] for i in 1:na ]
  S = [ res.values[Symbol("S[$i, $j]")] for i in 1:na for j in 1:ny ]
  N = reshape( S, na, ny )

=#



--- stop --- 

Type issues in VI and MCMC ...model structures need to be revised 
 
--- stop --- 


# Generate a Variational Inference estimate...  Not working ? 2024-10-26 
using Turing: Variational
advi = ADVI(10, 1000, AutoForwardDiff() ) # samples_per_step (gradient estimation), max_iters (no gradient steps)
res = vi(fmod, advi )

n_samples = 100
samples = rand(res, n_samples);

# extraction:
Fp = [ res.values[Symbol("Fp[$i]")] for i in 1:ny ]
Ft_hat = [ res.values[Symbol("Ft[$i]")] for i in 1:na ]
q = [ res.values[Symbol("q[$i]")] for i in 1:na ]
S =  [ res.values[Symbol("S[$i, $j]")] for i in 1:na for j in 1:ny ]

N = reshape( S, na, ny )

F = compute_fishing_mortality!(F, N, M; iplus=iplus)  # update F with finalized N's
Fall, fsubset, pl = summary_fishing_mortality(F, subset=1:5)
pl

B = N .* body_weight_kg[:,1]
Bmat, Btot, Bssb, pl = compute_biomass( B, maturity_ogive )
pl

Cpred = catch_cohort_estimate( N, F, M ) # Note some negative predicted catches 



# MCMC 

## Prior
n_samples = 100
prior = sample( fmod, Prior(), n_samples )  
showall(prior)

## SMC 
n_samples = 100
res = sample( fmod, Turing.SMC(),  n_samples   ) # sample in parallel


## NUTS
n_adapts, n_samples, n_chains, thin = 1000, 4000, 4, 0
target_acceptance_rate, max_depth, init_ϵ = 0.65, 8, 0.01

turing_sampler = Turing.NUTS( n_samples, target_acceptance_rate; max_depth=max_depth, init_ϵ=init_ϵ )

turing_sampler = Turing.NUTS(; adtype=AutoForwardDiff())

res = sample( fmod, turing_sampler,  n_samples   ) # sample in parallel

  # extraction:

    k = rand(1:nsims)  # nsims
    l = rand(1:nchains) # nchains
    k=1:n_samples
    l=1

    Fp = [ res[k, Symbol("Fp[$i]"), l] for i in 1:ny ]
    Ft = [ res[k, Symbol("Ft[$i]"), l] for i in 1:na ]
    q = [ res[k, Symbol("q[$i]"), l] for i in 1:na ]
    S =  [ res[k, Symbol("S[$i, $j]"), l] for i in 1:na for j in 1:ny ]

    S = vec( mean( hcat( S...), dims=1))

    N = reshape( S, na, ny )

    F = compute_fishing_mortality!(F, N, M; iplus=iplus)  # update F with finalized N's
    Fall, fsubset, pl = summary_fishing_mortality(F, subset=1:5)
    pl

    B = N .* body_weight_kg[:,1]
    Bmat, Btot, Bssb, pl = compute_biomass( B, maturity_ogive )
    pl

    Cpred = catch_cohort_estimate( N, F, M ) # Note some negative predicted catches 

   

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

  