## PyProb: A tool for inverting scientific simulators


PyProb is a [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming) system focused on scientific simulators and high-performance computing (HPC). 

!!! Note

    This documentation is intended for this branch of this fork: https://github.com/ammunk/pyprob/tree/andreas-dev/examples


### What it's for

Use PyProb if you have an existing simulator (in C, C++, Python, or [another language](/#supported-languages)) and you want to run it backwards, to infer the distribution of unknown quantities from experimental data.

For example, suppose you have a simulator that, given the mass of a particle, predicts its trajectory. 


```mermaid
flowchart LR
    B(Unknown variables) -->|simulator|A(simulated data)
```

You have an actual observed trajectory, gathered from experiment, and wish to *infer* the mass of the particle. In a sense, what you want to do is to run the simulator backwards.

```mermaid
flowchart RL
  A(real data)   -->|inference|B(Unknown variables)
```

This inversion is fundamentally a Bayesian problem, and the field of probabilistic programming languages (PPLs) has developed methods both to specify models (such as simulators) as programs, and to perform fast, automatic inference.

PyProb is one such PPL, with a focus on amortized inference. This is the approach of learning a neural network to make good guesses of the posterior distribution (given observations), which can then be used as proposals in a standard algorithm, like importance sampling.

**PyProb focuses on automatically specifying and training this neural network, as well as being able to interface with existing simulators in a variety of languages**. These simulators don't need to be differentiable, and they can have stochastic branching, or even a variable number of random choices.

[Get Started](/get_started){ .md-button .md-button--primary }


### How it works

You start with a simulator of your choosing in a [supported language](/#supported-languages).

For example, consider this very simple simulator in C++:

```cpp
#include <pyprob_cpp.h>

xt::xarray<double> forward()
{
  auto prior_mean = 1;
  auto prior_stddev = std::sqrt(5);
  auto likelihood_stddev = std::sqrt(2);

  auto prior = pyprob_cpp::distributions::Normal(prior_mean, prior_stddev);
  auto mu = pyprob_cpp::sample(prior);

  auto likelihood = pyprob_cpp::distributions::Normal(mu, likelihood_stddev);
  pyprob_cpp::observe(likelihood, "obs");

  return mu;
}
```

Here, the lines with `pyprob_cpp` are using the PyProb_cpp library, and replace what were originally calls to an RNG in your program. Understand this as describing a model with prior:
 
$$\mu \sim Normal(1, sqrt(5))$$ 

and likelihood:

$$y \sim Normal(\mu, sqrt(2))$$

You start a server like so:

```cpp
int main(int argc, char *argv[])
{
  // Extract the inter-process communication address from the arguments.
  auto serverAddress = (argc > 1) ? argv[1] : "ipc://@my_test";
  // Instantiate the model object
  pyprob_cpp::Model model = pyprob_cpp::Model(forward, "Gaussian with unknown mean C++");
  // Start running the simulator
  model.startServer(serverAddress);
  return 0;
}
```

Then in Python you construct a model object that calls this C++ executable. For a complete example, see: https://github.com/plai-group/covid/tree/master/FRED/tests/pyprob_cpp.


### Supported languages

We support front ends in the following languages through the [PPX](https://github.com/pyprob/ppx) interface, although beyond C++ and Python, you will have to generate bindings (to generate the equivalent of PyProb_cpp).

- C++
- Python
- C#
- Dart
- Go
- Java
- Kotlin
- Lobster
- Lua
- PHP
- Rust
- Swift
- TypeScript

### Supported inference methods

PyProb currently provides the following inference methods:

* Markov chain Monte Carlo
  * Lightweight Metropolis Hastings (LMH)
  * Random-walk Metropolis Hastings (RMH)
* Importance sampling
  * Regular sequential importance sampling (proposals from prior)
  * Inference compilation[^1]

[^1]: Inference compilation is an amortized inference technique for performing fast
repeated inference using deep neural networks to parameterize proposal
distributions for importance sampling. 

Inference compilation is the main focus of PyProb.


## License

PyProb is distributed under the BSD License.

