# Logistic discrete models

## Intuition

Population dynamics can be studied in many ways. Arguably, the simplest is to see it as a single system, such as the total number of individuals. Such single component models have a long tradition (Verhulst 1845, McKendrick & Pai 1912, Pearl & Reed 1920, Lotka 1925, Bacaër 2011). This is often called a **phenomenological** perspective in that it abstracts a multitude of processes and interactions into very few externally facing parameters and one state variable (population size). Often used for such cases, the logistic model specifies dynamics with two parameters: a maximum specific rate of change, and some upper limit of system state. Their utility and properties are well known and used in many domains.

Verhulst's original formulation of the logistic equation was used to model Belgium's population growth (Verhulst 1845, Bacaër 2011). His arguments were also phenomenological, alluding to Malthusian geometric (exponential) growth with some upper limit. The intrinsic rate of increase (r) represented the exponential rate of increase, while carrying capacity (K) represented this upper limit. Subsequently, McKendrick & Pai (1912) used it for bacterial growth and then (Pearl & Reed 1920, Lotka 1925) used it for the US population. It continues to be used today in numerous fields due to its underlying sigmoid shape (Beverton & Holt 1957, Pianka 1970, Bohner & Chieochan 2013).





## Taylor/MacLauran Series approximation

Looking for a stronger justification, Lotka (1925) argued that the logistic equation can be seen as a 2nd order Taylor (Maclaurin) series approximation of any general function $\frac{dN}{dt}=G(N)$:

$$\frac{dN}{dt} = rN -(rK^{-1}) N^{2} + O(N^{h}),$$  

where, $O(N^{h})$ are higher order terms, $h>2.$

## Linear first order processes at equilibrium

Pianka (1970, Fig. 9.5) suggested a slightly different approach. He begins with birth rate $\beta = \beta_0 - eN$ and death rate $\delta = \delta_0 + fN$ that are each linear functions of abundance. At equilibrium $N=K$, the lines intersect with $\beta = \delta$: 

$$\beta_0 - eK = \delta_0 + fK$$

$$r_0 = \beta_0 - \delta_0 = (e-f)K.$$

Away from equilibrium, after some algebra, we obtain the logistic equation:

$$\frac{dN}{dt} = rN = [(\beta_0 - eN) - (\delta_0 + fN)] N = r_0 N - r_0 N^2.$$

## Beverton-Holt 

Interestingly, the sigmoid Beverton-Holt difference equation has also been shown to be related to the logistic equation (Beverton & Holt 1957, Bohner & Chieochan 2013):

$$n_{t+1} = \frac {\nu Kn_t} { K_t + (\nu-1) n_t}.$$

It is discrete in time and also sigmoid in shape with asymptotic limit (carrying capacity that varies with time): $K_{t} > 0$ and growth rate $\nu>1$, for number $n_{t}>0$. 

If we define: $r= (\nu-1) / \nu$, then $\nu = 1/(1-r)$, 

then substitution and solving for $n_{t}$ and then ${\Delta n}_{t}$ gives:

$${\Delta n}_{t} = r n_{t+1} (K_{t} - n_{t+1}) / (K_{t}-rn_{t+1}),$$

which simplifies further to the discrete logistic difference equation:

$${\Delta n}_{t} = r n_{t+1} (1 - n_{t} / K_{t} ).$$ 

In the limit as $\Delta t\rightarrow0$, this becomes the differential form of the logistic equation:

$$\frac{dn}{dt} = r n (1 -n/K_{t} ).$$


## Logistic discrete map

The "discrete logistic map" is often represented as a Euler approximation of the logistic differential equation:

$${\frac{{dZ}}{{dt}}=r}Z{({{1-Z}K^{-1}})}.$$

Normalization of $Z$ to $z={Z/K}$ allows simplification to:

$$\frac{dz}{dt} = r z ( 1-z).$$ 

The Euler approximation for a small increment of time, $\tau_{t+1}={\tau_{t} + \Delta} \tau$, then $z$ in the next time interval is estimated as:

$$z_{t+1} = z_{t} + \Delta \tau \cdot \frac{dz}{dt} z_{t},$$ 

which upon substitution of the differential equation and simplification gives:

$$z_{t+1} = ( 1 + \Delta \tau r) z_{t} - \Delta \tau r z_{t}^{2}.$$

The identities: $\rho = 1 + \Delta \tau r$ and $n_{t}=\frac{\Delta\tau rz_{t}}{\rho}$ help to simplify further:

$$n_{t+1} = \frac{ \Delta \tau r z_{t+1}}{\rho}.$$ 

Substitution of the identity of $z_{t+1}$ to gives: 

$$n_{t+1} = \frac{\Delta \tau r (( 1 + \Delta \tau r) z_{t}-\Delta \tau r{z_t}^2)}{\rho} = \rho n_{t} (1-n_{t}).$$

Note that $\rho$ is ${1+\textnormal{rescaled}}{(r)}$, which means a fractional change relative to the previous *time step*. Similarly, $n$ are rescaled values of $z$, more meaningful with a time step of $\Delta t$. Importantly, this derivation is for infinitesimal changes in time. If the increment in time is relatively large, such as $\Delta t = 1 \; \textnormal{year}$, as it is commonly used in fisheries applications, the "infinitesimal" change assumption is not biologically reasonable.



## Surplus production

In the fisheries literature, the logistic model is also used from the perspective of surplus production. They treat the overall population **phenomenologically** as one homogenous entity (biomass, and not number though it could be converted if assumptions of average size can be made) with a simple perspective: increase and decrease are governed as a function of input-output processes that when combined provide some empirical "surplus production" which can be of various functional forms. 

It is called "surplus production" as the historical perspective was that a population generally produces more than is necessary to maintain/replace itself (due to natural mortality). This "surplus production" was thought to be safely diverted to human consumption/exploitation/harvest. From this perspective, maximum surplus production was the target, commonly known as "Maximum Sustainable Yield" (MSY).   

In terms of basic processes, this approach begins with mass balance:

>>   biomass(t+1) = biomass(t) + recruitment(t) + growth(t) - catch(t) - natural mortality(t).

Usually, growth and reproduction are grouped together as "production", and, "surplus production" = production - natural mortality, that is, in the absence of fishing. Ignored of course are respiration, movement, etc. which are absorbed by "surplus production". 

Historically, this is the amount that was thought that could be caught while still maintaining biomass at equilibrium:

$$B_{t+1} = B_t + S(B_t) - C_t,$$

where the functional form of the "surplus production", $S(B_t)$, is some (invariant) function of biomass, determined from observation. If,

- $S(B) = rB (1-B/K)$ ; then it is a "Schaefer" 1954 form

- $S(B) = rB(1-log(B)/log(K))$ ; then it is a "Fox" 1970 form

- $S(B) = \frac{r B(1-(B/K)^p)}{p}$ ; then it is known as a "Pella-Tomlinson" 1969 form. 

Note that the Pella-Tomlinson form == Schaefer form when $p=1$
and the Pella-Tomlinson form $\rightarrow$ Fox form as $p \rightarrow 0$.


## Methods of parameter estimation

Reference: 
 
Estimation has been attempted by various methods:

    - equilibrium assumptions (unstable): example ... 

    - regression (Schnute 1977; erratic)
    
        - log( O[t+1]/O[t] ) = 4 * m / K - 4 * m / (q *K^2) *(O[t] +O[t+1])/2 - q*( C[t]/O[t] ); O=observed biomass (index)

    - observation error or "time-series method" (aka, state space models) Hilborn & Walters (1992) - Chapter 8

        - O ~ q B 

All approaches have issues with stability, especially when used with least-squares or unconditioned Maximum Likelihood methods.

The current state of the art is to use Bayesian state-space methods to help stabilize and rationalize parameter estimates. The native Julia implementation of these models is provided in `AssessmentTools.BiomassDynamics` and visual diagnostics via `AssessmentTools.plot_state_variables` and `plot_fishery_predictions`:

```{julia}
using AssessmentTools
using AssessmentTools.BiomassDynamics

# Simulate forward logistic dynamics with observational noise
years = collect(2000.0:2024.0)
catches = fill(150.0, length(years))
sim = simulate_logistic_discrete(years, catches; r = 0.35, K = 2500.0, b0 = 1800.0, q = 1.0)

# Visual state assessment
plot_state_variables(sim)
```
 

### References

Bacaër N (2011) Verhulst and the logistic equation (1838). In: *A Short History of Mathematical Population Dynamics*. Bacaër N (ed), Springer, London, p 35--39.

Bertalanffy L von (1950) The theory of open systems in physics and biology. Science 111:23--29.

Beverton R, Holt S (1957) On the dynamics of exploited fish populations, Volume 19 of Fishery investigations (Great Britain, Ministry of Agriculture, Fisheries, and Food). HM Station Off Lond.

Bohner M, Chieochan R (2013) The Beverton--Holt q-difference equation. J Biol Dyn 7:86--95. 

Bolker BM (2008) Ecological Models and Data in R. Princeton University Press.

Lotka AJ (1920) Analytical note on certain rhythmic relations in organic systems. Proc Natl Acad Sci 6:410--415. 

Lotka AJ (1925) The elements of physical biology. Williams and Wilkins.

McKendrick AG, Pai MK (1912) XLV.---The Rate of Multiplication of
Micro-organisms: A Mathematical Study. Proc R Soc Edinb
31:649--655.
 
Pearl R, Reed LJ (1920) On the Rate of Growth of the Population of the United States since 1790 and Its Mathematical Representation. Proc Natl Acad Sci U S A 6:275--288.

Pianka ER (1970) On r- and K-selection. Am Nat 104:592--597.

Quinn TJ (2003) Ruminations on the Development and Future of Population Dynamics Models in Fisheries. Nat Resour Model 16:341--392.

Schaefer MB (1954) Some Aspects of the Dynamics of Populations Important to the Management of the Commercial Marine Fisheries.

Verhulst PF (1845) Resherches mathematiques sur la loi d'accroissement de la population. Nouv Memoires Acad R Sci 18:1--41.

Wangersky PJ (1978) Lotka-volterra population models. Annu Rev Ecol Syst 9:189--218.



 