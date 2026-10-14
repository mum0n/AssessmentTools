---
title: "Snow Crab, Scotian Shelf, Canada (NAFO Div. 4VWX) in 2023"
subtitle: "DDE model solutions"
author: "Snow Crab Group"
# author: "Jae S. Choi"
# footnote: "jae.choi@dfo-mpo.gc.ca"
institute: "Bedford Institute of Oceanography, DFO Science"
# date: "`r format(Sys.time(), '%d %B, %Y')`"
output:
  beamer_presentation:
    theme: "metropolis"
    colortheme: "seagull"
    fonttheme: "professionalfonts"
    fig_caption: yes
    # latex_engine: pdflatex
    latex_engine: lualatex 
    keep_tex: true
classoption: 
  - aspectratio=169 #16:9 wide
  - t  # top align
header-includes: 
  - \usepackage{graphicx}
  - \usepackage[font={scriptsize}, labelfont={bf}]{caption}
  # - \usepackage{float}
  # - \usepackage{subfig}
  # - \newcommand{\btiny}{\begin{tiny}}
  # - \newcommand{\etiny}{\end{tiny}}
params:
  year.assessment: 2023
  media_loc: "media"
  debugging: FALSE
  loc_dde: "/home/jae/bio.data/bio.snowcrab/fishery_model/2023/size_structured_dde_normalized"
--- 


<!-- Preamble


This is a Markdown document ... To create HTML or PDF, etc, run: 


  make quarto FN=dde YR=2023 SOURCE=~/projects/bio.snowcrab/inst/markdown WK=~/bio.data/bio.snowcrab/assessments # {via Quarto}

  make rmarkdown FN=dde YR=2023 SOURCE=~/projects/bio.snowcrab/inst/markdown WK=~/bio.data/bio.snowcrab/assessments {via Rmarkdown}

  make pdf FN=dde  # {via pandoc}

Alter year and directories to reflect setup or copy Makefile and alter defaults to your needs.
  
 

Huge > huge > LARGE > Large > large > normalsize > small > footnotesize > scriptsize > tiny


::: columns 
:::: column 

::::
 
:::: column

::::
:::


-->



<!-- Set up R-environment -->

```{r setup, include=FALSE}
  require(knitr)
  knitr::opts_chunk$set(
    root.dir = data_root,
    echo = FALSE,
    out.width="6.2in",
#     dev.args = list(type = "cairo"),
    fig.retina = 2,
    dpi=192
  )

  # inits and data loading (front load all required data)

  require(aegis)
  
  year.assessment = params$year.assessment
  year_previous = year.assessment - 1
  p = bio.snowcrab::load.environment( year.assessment=year.assessment )
  SCD = project.datadirectory("bio.snowcrab")
  media_loc = params$media_loc
  
  # fishery_model_results = file.path( "/home", "jae", "projects", "model_fishery", "outputs" )
  fishery_model_results = file.path( SCD, "fishery_model" )

  sn_env = dde_results( year.assessment, debugging=params$debugging, loc_dde=params$loc_dde, return_as_list=TRUE  ) 

  attach(sn_env)
 
  require(data.table) # for speed
  require(lubridate)
  require(stringr) 
  require(gt)  # table formatting
  library(janitor)
  require(ggplot2)
  require(aegis) # map-related 
  require(bio.taxonomy)  # handle species codes
 
 
  if (0) {
    # debugging
    params = list(
      year.assessment=2023, 
      media_loc= "~/bio.data/bio.snowcrab/assessments/media",
      debugging= FALSE,
      loc_dde= "~/bio.data/bio.snowcrab/fishery_model/2023/size_structured_dde_normalized"
    )
  }


```


To provide an assessment of the:

  1. status of Snow Crabs
  2. relative abundance and exploitation rates
  3. evaluate consequences of different harvest levels upon abundance and exploitation rates
  4. bycatch in survey


Amendments of the Fisheries Act in 2022 (*Fish Stock Provisions*), have encoded management approaches and requirements relative to biological reference points. In the SSE Snow Crab Fishery, these biological reference points were originally defined heuristically with a  *phenomenological* model (a descriptive, model without mechanism, herein "Model 1"; Choi 2023), with all participants to the management process being aware that it was simplistic at best and that it served as supplemental information to help provide context rather than be the tool used for management. As there is now a strong legal requirement that such reference points become a primary management tool, there is a need to define these reference point with greater care and evaluate their robustness and utility with other models. Here, we will provide some additional information towards this goal and present "Model 2" results (see Ecosystem considerations, below and Choi 2023), in order to obtain more context around the reference points and status (objectives 2 and 3 of the Terms of Reference).

  
   
## Stock status:  Model 2  {.c}

- Embed habitat viability variations directly into a model

```{r dde-predictions-everything,  fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Modelled predictions of fishable biomass of Snow Crab (Model 2). N-ENS (left), S-ENS (middle), and 4X (right). Green: no fishing; orange: with fishing; gray: biomass index; and dark lines are post-fishery, posterior averages.'  }
  odir = file.path( fishery_model_results, year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( odir, "plot_predictions_everything_cfanorth.pdf" ) 
  fn2 = file.path( odir, "plot_predictions_everything_cfasouth.pdf" ) 
  fn3 = file.path( odir, "plot_predictions_everything_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-predictions-everything)
``` 
 
 

## Viable habitat in a stage-dependent model

A complementary modelling approach ("Model 2") has been developed Choi (2023). It is a six-component system (five male stages from instar 9 [40.9-55.1mm], 10 [55.1-74.4mm], 11 [74.4-95mm], immature [95mm+], mature [95mm+] and mature females) delay differential model, where each component has a molt and/or birth from a previous stage in balance with a constant (first order) background death rate; a second death rate associated with varying levels of viable habitat (second order, similar to the logistic form); and fishing mortality, assumed to be known without error. Model 2 suggests population trends that are mostly comparable with Model 1 (Figure \@ref(fig:dde-predictions-everything)). However, overall biomass is estimated to be lower which results in fishing mortality estimates being higher (Figure \@ref(fig:dde-fishing-mortality)). 

 
 
```{r dde-predictions-everything,  fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Modelled predictions of fishable biomass of Snow Crab (Model 2). N-ENS (left), S-ENS (middle), and 4X (right). Green: no fishing; orange: with fishing; gray: biomass index; and dark lines are post-fishery, posterior averages.'  }
  fn1 = file.path( loc_dde, "plot_predictions_everything_cfanorth.pdf" ) 
  fn2 = file.path( loc_dde, "plot_predictions_everything_cfasouth.pdf" ) 
  fn3 = file.path( loc_dde, "plot_predictions_everything_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-predictions-everything)
``` 
 
   
```{r dde-fishing-mortality, fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Modelled fishing mortality of Snow Crab (Model 2) for N-ENS (left), S-ENS (middle), and 4X (bottom).' }
  fn1 = file.path( loc_dde, "plot_fishing_mortality_cfanorth.pdf" ) 
  fn2 = file.path( loc_dde, "plot_fishing_mortality_cfasouth.pdf" ) 
  fn3 = file.path( loc_dde, "plot_fishing_mortality_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-fishing-mortality)
``` 
  

Fishing mortality estimates from Model 2 (Figure \@ref(fig:dde-fishing-mortality)) have an overall form that is similar to those of Model 1 (Figure \@ref(fig:logisticFishingMortality)). As with Model 1, Model 2 suggests peak fishing mortality likely occurred in N-ENS in 2005 and earlier. After strong and difficult reductions, it has slowly increased over time. In S-ENS, it has been more stable and conservative throughout the timeseries. In 4X, prior to formal assessments beginning in 2005, fishing mortality was likely to have been high, but has since declined to a range  consistent with the other regions. 

The important difference between the two models is that Model 2 suggests fishing mortality rates are higher in magnitude than those of Model 1. Model 2 estimates of fishing mortality rate for N-ENS in the `r year.assessment` was `r round(ddeFM_north[t0],2)` (annual exploitation rate of `r round(100*(exp(ddeFM_north[t0])-1),2)`%), up from the `r year_previous` rate of `r round(ddeFM_north[t1],1)` (annual exploitation rate of `r round(100*(exp(ddeFM_north[t1])-1),1)`%). In S-ENS, the `r year.assessment` fishing mortality was `r round(ddeFM_south[t0],2)` (annual exploitation rate of `r round(100*(exp(ddeFM_south[t0])-1),1)`%), down slightly from the `r year_previous` rate of `r round(ddeFM_south[t1],2)` (annual exploitation rate of `r round(100*(exp(ddeFM_south[t1])-1),1)`%). In 4X, the `r year.assessment`-`r year.assessment+1` season (ongoing) fishing mortality was `r round(ddeFM_4x[t0],2)` (annual exploitation rate of `r round(100*(exp(ddeFM_4x[t0])-1),1)`%), down from the `r year_previous`-`r year.assessment` season rate of `r round(ddeFM_4x[t1],2)` (annual exploitation rate of `r round(100*(exp(ddeFM_4x[t1])-1),1)`%). 

Model 2 does not have a concept of FMSY as there is no stable solution to such an externally perturbed system (viable habitat). The FMSY from Model 1 in theory should be approximately 0.5. If this is accepted, then we can see that N-ENS was close to this threshold in 2022. The stock status can be seen in an equivalent format as for Model 1 (Figure \@ref(fig:dde-hcr)).  



  

Fishery footprint (Figures \@ref(fig:dde-footprint-trace)) represents the change in abundance between the orange lines (Figure \@ref(fig:dde-predictions-everything)) where fishing occurred, with the green lines: the abundance that might have been if fishing did not occur. Fishery footprint has been to depress potential abundance by  25% to 50% in N-ENS since 2010. In S-ENS, the fisheries footprint has been stable between 10% to 30%. Area 4X footprint has declined from over 50% in the pre-2010 period to now a value of less than 25%. The fisheries footprint of all areas are now in a stable and robustly precautionary state, with the exception of N-ENS. 

The current estimates of some key parameters derived from the six-component model are shown in Table 7. The estimate of carrying capacity is different.  As with Model 1, N-ENS and S-ENS are still considered in the "healthy" zone. However, Model 2 suggests that 4X may still be in the "healthy" zone, though still close to the "cautious" zone.  Reality is likely somewhere in between the two Models representations.
 



*Table 7. Reference points from the six-component fishery model (Model 2): K is Carrying capacity (numbers converted to biomass, kt, using average body weight of the fishable component); and b is "birth" rate of mature females (non-dimensional).*


|       |  $K$ [SD] | $b$ [SD] |
| :---: |    :----:   | :---: |
| N-ENS | `r Kdde_north` [`r Kdde_north_sd`] | `r bdde2_north` [`r bdde2_north_sd`] |
| S-ENS | `r Kdde_south` [`r Kdde_south_sd`] | `r bdde2_south` [`r bdde2_south_sd`] |
| 4X    | `r Kdde_4x`   [`r Kdde_4x_sd`] | `r bdde2_4x` [`r bdde2_4x_sd`] |



Forward projections at different levels of harvest relative to 2022 TACs suggest that all areas can expect some reduction in abundance (Figure \@ref(fig:projections-fb)) and an increase in the fishery footprint (Figure \@ref(fig:projections-footprint)). In N-ENS, all scenarios (80%, 100%, 120% of `r year.assessment`'s TAC, suggest fishery footprint is at risk of exceeding 50%. 

These results from Model 2 are supplementary in nature to provide additional context. Their direct use in defining reference point thresholds is **not recommended** at this point. 




--- alt text

In light of these issues, we present the results of another complementary modelling approach: a *Delay Differential Stage Structured Model* (**Model 2**). The rationale for Model 2 is developed and described in detail elsewhere (Choi 2023). It is a six-component system (five male stages from instar 9 [40.9-55.1mm], 10 [55.1-74.4mm], 11 [74.4-95mm], immature [95mm+], mature [95mm+] and mature females) delay differential continuous model, where each component has a molt and/or birth from a previous stage in balance with a constant (first order) background death rate; a second-order death rate associated with varying levels of viable habitat, similar to the logistic form; and fishing mortality, assumed to be known without error. 

It must be emphasized that Model 2, though promising, is still in the *early stages of development*. Most notably, there are many more processes (parameters) being estimated; the more complex parameter space results in many local-optima. Though the use of the NUTS sampler in a Bayesian context permits some measure of robustness from becoming stuck in such local-optima, this slows down computations. Further, missing from the model are: predation mortality and movement. These latter processes will need to be parameterized for the model system to be more robust and complete. However, as it is already computationally quite expensive taking up to several days to complete one area, additional model complexity will need to be added carefully.

As Model 2 is continuous in form, we can infer greater detail with respects to the time evolution and in particular the intensity of fishing vs the usually slower recovery that depends upon recruitment levels. This creates the sawtooth pattern seen in Figure (\@ref(fig:dde-predictions-everything), orange lines). This also demonstrates why it is so difficult to model the dynamics of snow crab as temporal aliasing can be large. Even small changes in the timing of surveys or fishing can alter our understanding of the dynamics of the population. Further, the volatility of the environment in conjunction with the small population size (especially in 4X), renders a robust understanding of population dynamics a significant challenge.

Note also, the diminished overall magnitudes of the biomass (Tables 6, 7) as the relative scale of fisheries activity, variability in viable habitat, and dynamics between components informs Model 2 to more reasonable bounds. Model 2 still had difficulty tracking the fishable biomass index in N-ENS (2016 to present), S-ENS (all years) and 4X (2016-2022). There are other factors that are not captured by the model and/or the data. We hypothesize that one such process is subarea dynamics with sub-legal components that are out of phase and internal structure (inshore-offshore and southwest-northeast decoherence) causing an overall dampening effect. Predation and movement are the missing processes. But the survey itself taking up to 4 months to complete, can be seen to be problematic as well, in that temporal and spatial aliasing is introduced.

Fishing mortality estimates from Model 2 (Figure \@ref(fig:dde-fishing-mortality)) have an overall form that is similar to those of Model 1 (Figure \@ref(fig:logisticFishingMortality)). As with Model 1, Model 2 suggests peak fishing mortality likely occurred in N-ENS in 2005 and earlier. After strong and difficult reductions, it has slowly increased over time. In S-ENS, it has been more stable and conservative throughout the timeseries. In 4X, prior to formal assessments beginning in 2005, fishing mortality was likely to have been high, but has since declined to a range consistent with the other regions. 

The important difference between the two models is that Model 2 suggests fishing mortality rates are higher in magnitude than those of Model 1. Model 2 estimates of fishing mortality rate for N-ENS in the `r year.assessment` was `r round(ddeFM_north[t0],2)` (annual exploitation rate of `r round(100*(exp(ddeFM_north[t0])-1),2)`%), up from the `r year_previous` rate of `r round(ddeFM_north[t1],1)` (annual exploitation rate of `r round(100*(exp(ddeFM_north[t1])-1),1)`%). In S-ENS, the `r year.assessment` fishing mortality was `r round(ddeFM_south[t0],2)` (annual exploitation rate of `r round(100*(exp(ddeFM_south[t0])-1),1)`%), down slightly from the `r year_previous` rate of `r round(ddeFM_south[t1],2)` (annual exploitation rate of `r round(100*(exp(ddeFM_south[t1])-1),1)`%). In 4X, the `r year.assessment`-`r year.assessment+1` season (ongoing) fishing mortality was `r round(ddeFM_4x[t0],2)` (annual exploitation rate of `r round(100*(exp(ddeFM_4x[t0])-1),1)`%), down from the `r year_previous`-`r year.assessment` season rate of `r round(ddeFM_4x[t1],2)` (annual exploitation rate of `r round(100*(exp(ddeFM_4x[t1])-1),1)`%). 

There is no concept of FMSY in Model 2 as there is no stable solution to such an externally perturbed system (viable habitat). One can, however, accept Model 1's assertion that FMSY should be somewhere close to 0.5. Using this an approximate landmark, N-ENS may have come close to this threshold in 2022. The trajectory of stock status (Figure \@ref(fig:dde-hcr)), would suggest that all areas are in "healthy" states, though 4X is very close to the border of the "cautious" zone. The important difference in interpretation is that carrying capacity estimates may be generally lower. In particular, for S-ENS, dynamics may have been moving quite close to carrying capacity levels and sometimes even surpassing it when good environmental conditions are followed rapidly by average conditions. Overall, reality is likely somewhere in between the two Models inferences.
 
Further, assuming there is *no fishing* in the next year, there is a prediction of increased abundance for N-ENS and S-ENS and a *decrease* in abundance in 4X (Figure (\@ref(fig:dde-predictions-everything)), orange lines after 2022 converge towards the green). In contrast, Model 1 will *always* project an increase due to its naive assumptions (\@ref(fig:logisticPredictions)).  

An alternative, and perhaps more intuitive index of fishing effect is to express it not in terms of an instantaneous rate nor interval approximations which are bound to parameters that can be biased (such as K, r, FMSY), but rather, how much fishing has moved the dynamics of a population away from what it may have been if there had been no fishing. This **fishery footprint** (Figure \@ref(fig:dde-footprint-trace)) represented by the degree of divergence between the orange lines (with fishing) and the green lines (no fishing). This fishery footprint has been about 25% to 50% in N-ENS since 2010. In S-ENS, the fisheries footprint has been stable and low between 10% to 30%. Area 4X fisheries footprint has declined from over 50% in the pre-2010 period to now a value of less than 25%. This suggests fisheries exploitation in 2022 was reasonably precautionary, with the exception of N-ENS.  
 
By assuming that:

  - The posterior distributions of these parameters are robust
  - Fishing pattern across time will be similar to the past five years
  - Average size of the fishable component will be similar
  - Habitat viability will be similar to the current year
  - Predation and prey conditions will be similar

we can project slightly more mechanistic expectations of future time trends for the fishable component (Figure \@ref(fig:projections-fb)) for varying levels of catch.  All scenarios considered (80%, 100% and 120% of `r year.assessment`'s TACs) lead to declining trends in the fishable biomass, with steeper declines with higher harvest scenarios.  

Exact probabilities could also be computed from the posterior distributions, but the approach would have a high risk of failure due to the large number of assumptions. Rather than presuming that we have captured the true dynamics and the associated uncertainties, we instead, focus upon the effects of harvest scenarios upon the fishery footprint, which is more robust to errors of parameterizations as it is internally coherent and so perhaps a more utilitarian device to assess sub-component dynamics and integrate them to judge the directionality the fishery at different levels of exploitation (fishery footprint; Figure \@ref(fig:projections-footprint)).

These results from Model 2 are supplementary in nature to provide additional context. Their direct use in defining reference point thresholds is **not recommended** at this point. 


Supplemental analysis using a more complex model (Model 2) supports these inferences. However, they suggest that Model 1's estimates of carrying capacity may be overly optimistic and as such fishing mortality estimates may be biased low. Fishery footprint, a sensitive index of how much a fishery activity is altering the dynamics of a population, is inferred to be lower than 50% in all areas. Assuming current ecosystem conditions persist, Model 2 suggest that a status quo TAC for 2023 would likely result in reductions in overall abundance and that the fishery footprint would go above 50% in N-ENS, upto 40% in S-ENS and in 4X. 

\clearpage

```{r dde-footprint-trace,  fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Fishery footprint derived from the Snow Crab Model 2.  N-ENS (left), S-ENS (middle), and 4X (right).' }
  fn1 = file.path( loc_dde, "plot_footprint_trace_cfanorth.pdf" ) 
  fn2 = file.path( loc_dde, "plot_footprint_trace_cfasouth.pdf" ) 
  fn3 = file.path( loc_dde, "plot_footprint_trace_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-footprint-trace)
``` 

 

```{r projections-fb, fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Forward projections derived from the Snow Crab Model 2 for each area (columns: N-ENS, S-ENS and 4X) and for different TACs (rows: 0.8, 1.0 and 1.2 X Status quo TAC).' }
  fn1 = file.path( loc_dde, "plot_trace_projections_cfanorth__0.8__.pdf" ) 
  fn2 = file.path( loc_dde, "plot_trace_projections_cfasouth__0.8__.pdf" ) 
  fn3 = file.path( loc_dde, "plot_trace_projections_cfa4x__0.8__.pdf" ) 
  fn4 = file.path( loc_dde, "plot_trace_projections_cfanorth__1.0__.pdf" ) 
  fn5 = file.path( loc_dde, "plot_trace_projections_cfasouth__1.0__.pdf" ) 
  fn6 = file.path( loc_dde, "plot_trace_projections_cfa4x__1.0__.pdf" ) 
  fn7 = file.path( loc_dde, "plot_trace_projections_cfanorth__1.2__.pdf" ) 
  fn8 = file.path( loc_dde, "plot_trace_projections_cfasouth__1.2__.pdf" ) 
  fn9 = file.path( loc_dde, "plot_trace_projections_cfa4x__1.2__.pdf" ) 
  include_graphics(c(fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9) )
  # \@ref(fig:projections-fb)
``` 
 


```{r projections-footprint, fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Forward projections derived from the Snow Crab Model 2 for each area (columns: N-ENS, S-ENS and 4X) and for different TACs (rows: 0.8, 1.0 and 1.2 X Status quo TAC).' }
  fn1 = file.path( loc_dde, "plot_trace_footprint_projections_cfanorth__0.8__.pdf" ) 
  fn2 = file.path( loc_dde, "plot_trace_footprint_projections_cfasouth__0.8__.pdf" ) 
  fn3 = file.path( loc_dde, "plot_trace_footprint_projections_cfa4x__0.8__.pdf" ) 
  fn4 = file.path( loc_dde, "plot_trace_footprint_projections_cfanorth__1.0__.pdf" ) 
  fn5 = file.path( loc_dde, "plot_trace_footprint_projections_cfasouth__1.0__.pdf" ) 
  fn6 = file.path( loc_dde, "plot_trace_footprint_projections_cfa4x__1.0__.pdf" ) 
  fn7 = file.path( loc_dde, "plot_trace_footprint_projections_cfanorth__1.2__.pdf" ) 
  fn8 = file.path( loc_dde, "plot_trace_footprint_projections_cfasouth__1.2__.pdf" ) 
  fn9 = file.path( loc_dde, "plot_trace_footprint_projections_cfa4x__1.2__.pdf" ) 
  include_graphics(c(fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9) )
  # \@ref(fig:projections-footprint)
``` 

\clearpage

```{r dde-hcr, fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Reference points derived from the Snow Crab Model 2 for N-ENS (left), S-ENS (middle), and 4X (bottom).' }
  fn1 = file.path( loc_dde, "plot_hcr_cfanorth.pdf" ) 
  fn2 = file.path( loc_dde, "plot_hcr_cfasouth.pdf" ) 
  fn3 = file.path( loc_dde, "plot_hcr_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-hcr)
``` 

## Stock status:  Fishery Footprint   {.c}
 
```{r dde-fisheryfootprint,  fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Fishery footprint (Model 2). N-ENS (left), S-ENS (middle), and 4X (right). Projections are based upon status quo TACs.'  }
  odir = file.path( fishery_model_results, year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( odir, "plot_trace_footprint_projections_cfanorth__1.0__.pdf" ) 
  fn2 = file.path( odir, "plot_trace_footprint_projections_cfasouth__1.0__.pdf" ) 
  fn3 = file.path( odir, "plot_trace_footprint_projections_cfa4x__1.0__.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-fisheryfootprint)
``` 
 
##  (N-ENS, S-ENS, 4X)


```{r dde-predictions,  out.width='32%', fig.show='hold', fig.align='center', fig.cap='predictions' }
  loc = file.path( homedir, "projects", "model_fishery",  "outputs",  year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( loc, "plot_predictions_cfanorth.pdf" ) 
  fn2 = file.path( loc, "plot_predictions_cfasouth.pdf" ) 
  fn3 = file.path( loc, "plot_predictions_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-predictions)
``` 
   
  
##  Fishery Footprint   {.c}
 
```{r dde-fisheryfootprint,  out.width='32%', fig.show='hold', fig.align='center', fig.cap='Fishery footprint (Model 2). N-ENS (left), S-ENS (middle), and 4X (right). Projections are based upon status quo TACs.'  }
  loc = file.path( homedir, "projects", "model_fishery", "outputs",  year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( loc, "plot_trace_footprint_projections_cfanorth__1.0__.pdf" ) 
  fn2 = file.path( loc, "plot_trace_footprint_projections_cfasouth__1.0__.pdf" ) 
  fn3 = file.path( loc, "plot_trace_footprint_projections_cfa4x__1.0__.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-fisheryfootprint)
``` 


##  (N-ENS, S-ENS, 4X)
```{r dde-hcr,  out.width='32%', fig.show='hold', fig.align='center', fig.cap='hcr' }
  loc = file.path( homedir, "projects", "model_fishery",  "outputs",  year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( loc, "plot_hcr_cfanorth.pdf" ) 
  fn2 = file.path( loc, "plot_hcr_cfasouth.pdf" ) 
  fn3 = file.path( loc, "plot_hcr_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-hcr)
```

##  (N-ENS, S-ENS, 4X)
```{r dde-footprint-trace,   out.width='32%', fig.show='hold', fig.align='center', fig.cap='footprint trace' }
  loc = file.path( homedir, "projects", "model_fishery", "outputs",  year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( loc, "plot_footprint_trace_cfanorth.pdf" )
  fn2 = file.path( loc, "plot_footprint_trace_cfasouth.pdf" )
  fn3 = file.path( loc, "plot_footprint_trace_cfa4x.pdf" )
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-footprint-trace)
``` 


##  (N-ENS, S-ENS, 4X)
```{r dde-hcr-footprint,  out.width='32%', fig.show='hold', fig.align='center', fig.cap='hcr footprint' }
  loc = file.path( homedir, "projects", "model_fishery", "outputs",  year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( loc, "plot_hcr_footprint_cfanorth.pdf" )
  fn2 = file.path( loc, "plot_hcr_footprint_cfasouth.pdf" )
  fn3 = file.path( loc, "plot_hcr_footprint_cfa4x.pdf" )
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-hcr-footprint)
```

##  (N-ENS, S-ENS, 4X)
```{r dde-predictions-trace, out.width='32%', fig.show='hold', fig.align='center', fig.cap="predictions_trace" }
 loc = file.path( homedir, "projects", "model_fishery", "outputs", year.assessment, "size_structured_dde_normalized" )
 include_graphics( file.path( loc, "plot_predictions_trace_cfanorth.pdf" ) )
 include_graphics( file.path( loc, "plot_predictions_trace_cfasouth.pdf" ) ) 
 include_graphics( file.path( loc, "plot_predictions_trace_cfa4x.pdf" ) )
```


 
```{r dde-predictions-everything,  fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Modelled predictions of fishable biomass of Snow Crab (Model 2). N-ENS (left), S-ENS (middle), and 4X (right). Green: no fishing; orange: with fishing; gray: biomass index; and dark lines are post-fishery, posterior averages.'  }
  odir = file.path( SCD, "fishery_model", year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( odir, "plot_predictions_everything_cfanorth.pdf" ) 
  fn2 = file.path( odir, "plot_predictions_everything_cfasouth.pdf" ) 
  fn3 = file.path( odir, "plot_predictions_everything_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-predictions-everything)
``` 
 

   
```{r dde-fishing-mortality, fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Modelled fishing mortality of Snow Crab (Model 2). N-ENS (left), S-ENS (middle), and 4X (right).' }
  odir = file.path( SCD, "fishery_model", year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( odir, "plot_fishing_mortality_cfanorth.pdf" ) 
  fn2 = file.path( odir, "plot_fishing_mortality_cfasouth.pdf" ) 
  fn3 = file.path( odir, "plot_fishing_mortality_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-fishing-mortality)
``` 
    
```{r dde-hcr, fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Reference points derived from the Snow Crab Model 2. N-ENS (left), S-ENS (middle), and 4X (right).' }
  odir = file.path( SCD, "fishery_model", year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( odir, "plot_hcr_cfanorth.pdf" ) 
  fn2 = file.path( odir, "plot_hcr_cfasouth.pdf" ) 
  fn3 = file.path( odir, "plot_hcr_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-hcr)
```  



```{r dde-hcr-footprint, fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Reference points  derived from the Snow Crab Model 2, using fishery footprint. N-ENS (left), S-ENS (middle), and 4X (right).' }
  odir = file.path( SCD, "fishery_model", year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( odir, "plot_hcr_footprint_cfanorth.pdf" ) 
  fn2 = file.path( odir, "plot_hcr_footprint_cfasouth.pdf" ) 
  fn3 = file.path( odir, "plot_hcr_footprint_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-hcr-footprint)
``` 


 
```{r dde-footprint-trace,  fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Fishery footprint derived from the Snow Crab Model 2. N-ENS (left), S-ENS (middle), and 4X (right).' }
  odir = file.path( SCD, "fishery_model", year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( odir, "plot_footprint_trace_cfanorth.pdf" ) 
  fn2 = file.path( odir, "plot_footprint_trace_cfasouth.pdf" ) 
  fn3 = file.path( odir, "plot_footprint_trace_cfa4x.pdf" ) 
  include_graphics(c(fn1, fn2, fn3) )
  # \@ref(fig:dde-footprint-trace)
``` 
 

```{r projections-fb, fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Forward projections derived from the Snow Crab Model 2 for each area (columns: N-ENS, S-ENS and 4X) and for different TACs (rows: 0.8, 1.0 and 1.2 X Status quo TAC).' }
  odir = file.path( SCD, "fishery_model", year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( odir, "plot_trace_projections_cfanorth__0.8__.pdf" ) 
  fn2 = file.path( odir, "plot_trace_projections_cfasouth__0.8__.pdf" ) 
  fn3 = file.path( odir, "plot_trace_projections_cfa4x__0.8__.pdf" ) 
  fn4 = file.path( odir, "plot_trace_projections_cfanorth__1.0__.pdf" ) 
  fn5 = file.path( odir, "plot_trace_projections_cfasouth__1.0__.pdf" ) 
  fn6 = file.path( odir, "plot_trace_projections_cfa4x__1.0__.pdf" ) 
  fn7 = file.path( odir, "plot_trace_projections_cfanorth__1.2__.pdf" ) 
  fn8 = file.path( odir, "plot_trace_projections_cfasouth__1.2__.pdf" ) 
  fn9 = file.path( odir, "plot_trace_projections_cfa4x__1.2__.pdf" ) 
  include_graphics(c(fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9) )
  # \@ref(fig:projections-fb)
``` 
 


```{r projections-footprint, fig.show='hold', out.width='32%', echo=FALSE, fig.align='center', fig.cap='Forward projections derived from the Snow Crab Model 2 for each area (columns: N-ENS, S-ENS and 4X) and for different TACs (rows: 0.8, 1.0 and 1.2 X Status quo TAC).' }
  odir = file.path( SCD, "fishery_model", year.assessment, "size_structured_dde_normalized" )
  fn1 = file.path( odir, "plot_trace_footprint_projections_cfanorth__0.8__.pdf" ) 
  fn2 = file.path( odir, "plot_trace_footprint_projections_cfasouth__0.8__.pdf" ) 
  fn3 = file.path( odir, "plot_trace_footprint_projections_cfa4x__0.8__.pdf" ) 
  fn4 = file.path( odir, "plot_trace_footprint_projections_cfanorth__1.0__.pdf" ) 
  fn5 = file.path( odir, "plot_trace_footprint_projections_cfasouth__1.0__.pdf" ) 
  fn6 = file.path( odir, "plot_trace_footprint_projections_cfa4x__1.0__.pdf" ) 
  fn7 = file.path( odir, "plot_trace_footprint_projections_cfanorth__1.2__.pdf" ) 
  fn8 = file.path( odir, "plot_trace_footprint_projections_cfasouth__1.2__.pdf" ) 
  fn9 = file.path( odir, "plot_trace_footprint_projections_cfa4x__1.2__.pdf" ) 
  include_graphics(c(fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9) )
  # \@ref(fig:projections-footprint)
``` 

## References

\begin{tiny}



Choi, Jae S. 2023. A Multi-Stage, Delay Differential Model of Snow Crab Population Dynamics in the Scotian Shelf of Atlantic Canada. bioRxiv. https://doi.org/10.1101/2023.02.13.528296.



\end{tiny}
 
# END  
 
