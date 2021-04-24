Algorithm
=========

Here we describe in some detail the XFaster algorithm and how it is derived.
The code implementation of this algorithm is described in :ref:`the tutorial<Tutorial>`.

XFaster is a hybrid of two types of estimators: a pseudo-:math:`C_\ell` Monte Carlo estimator a la MASTER and PolSPICE, and a quadratic estimator. The latter is the special sauce, so we'll go into some detail about it here.

First, we'll introduce the Newton method for iteratively finding the root of a function. There are helpful pictures to visualize how this method works `on wikipedia <https://en.wikipedia.org/wiki/Newton%27s_method>`_. Say you have some continuous function, :math:`f`, and your initial guess for its root is :math:`x_0`. We can Taylor expand :math:`f` around this guess like so:

.. math::
   f\left(x_{0}+\epsilon\right)=f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right) \epsilon+\frac{1}{2} f^{\prime \prime}\left(x_{0}\right) \epsilon^{2}+\ldots
   :label: eq1

And to first order,

.. math::
   f\left(x_{0}+\epsilon\right) \approx f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right) \epsilon
   :label: eq2

We set the left side to zero, and solve for the step size for the next iteration:

.. math::
   \epsilon_{0}=-\frac{f\left(x_{0}\right)}{f^{\prime}\left(x_{0}\right)}
   :label: eq3

We solve for :math:`f` and :math:`f'` at the new guess, and continue the iteration, where the forumula for each next guess of the root is:

.. math::
   x_{n+1}=x_{n}+\epsilon_{n}=x_{n}-\frac{f\left(x_{n}\right)}{f^{\prime}\left(x_{n}\right)}
   :label: eq4

A major caveat with using this method is that if the derivative goes to zero near the starting or final root guess, things blow up and it will not converge. However, if the function is well behaved, and the initial guess is close enough, it can be shown that this method converges quadratically.

Ultimately, XFaster isn't going to solve for a root; it's going to find the extreme of some function. So instead of finding where :math:`f=0`, we want to find where its first derivative goes to 0. So now we're going to take the first derivative of equation 1 and set it equal to 0:

.. math::
   0=\frac{d}{dx}\left(f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right) \epsilon_{0}+\frac{1}{2} f^{\prime \prime}\left(x_{0}\right) \epsilon_{0}^{2}\right)=f^{\prime}\left(x_{0}\right)+f^{\prime \prime}\left(x_{0}\right) \epsilon_{0}
   :label: eq5


where we're throwing out terms higher than second derivative. Now, our step size is given by

.. math::
   \epsilon_{0}=-\frac{f^{\prime}\left(x_{0}\right)}{f^{\prime \prime}\left(x_{0}\right)}
   :label: eq6

Extending this method to multiple variables, the first derivative becomes a gradient, and the second derivative becomes the Hessian:

.. math::
   \mathbf{H}=\left[\begin{array}{cccc}{\frac{\partial^{2} f}{\partial x_{1}^{2}}} & {\frac{\partial^{2} f}{\partial x_{1} \partial x_{2}}} & {\cdots} & {\frac{\partial^{2} f}{\partial x_{1} \partial x_{n}}} \\ {\frac{\partial^{2} f}{\partial x_{2} \partial x_{1}}} & {\frac{\partial^{2} f}{\partial x_{2}^{2}}} & {\cdots} & {\frac{\partial^{2} f}{\partial x_{2} \partial x_{n}}} \\ {\vdots} & {\vdots} & {\ddots} & {\vdots} \\ {\frac{\partial^{2} f}{\partial x_{n} \partial x_{1}}} & {\frac{\partial^{2} f}{\partial x_{n} \partial x_{2}}} & {\cdots} & {\frac{\partial^{2} f}{\partial x_{n}^{2}}}\end{array}\right]
   :label: eq7


for an expression for the local extremum of

.. math::
   x^1_{n}=x^0_{n}-\mathbf{H}^{-1} \nabla f\left(x^0_{n}\right)
   :label: step


Because it's costly to compute :math:`H` for each iteration, we can instead make the approximation of using its expectation value, which does not depend on the data. This is equivalent to the Fisher information matrix:

.. math::
   \mathcal{F}_{i j}=\left\langle\mathbf{H}_{i j}\right\rangle=\left\langle\frac{\partial^{2} f}{\partial x_{i} \partial x_{j}}\right\rangle
   :label: fish_approx

This has all so far been an abstract exercise in how to find the values of the variables that maximize some function that depends on them. Now let's get into what XFaster uses it for, maximizing the likelihood function, which we approximate to be Gaussian:

.. math::
   \mathcal{L}(\mathbf{d} | \theta)=\frac{1}{|2 \pi \mathbf{C}|^{1 / 2}} \exp \left(-\frac{1}{2} \mathbf{d} \cdot \mathbf{C}^{-1} \cdot \mathbf{d}^{T}\right)
   :label: eqn10

where :math:`\mathbf{d}` is an observed data set, :math:`\theta` are the model parameters, and :math:`\mathbf{C}` is the covariance matrix, which depends on the model parameters: :math:`\textbf{C}(\theta)=\textbf{S}(\theta)+\textbf{N}`, where :math:`\textbf{S}` is signal and :math:`\textbf{N}` is noise.

For XFaster, our parameters, :math:`\theta` that will be fit to the data are the bandpowers, :math:`\mathcal{C}_\ell`. We want to maximize the log likelihood (so we can take derivatives more easily and since it is maximized where the likelihood is maximized), so we can use Equation :math:numref:`step` and the Fisher approximation of Equation :math:numref:`fish_approx` to write down the size of the step we need from our initial bandpower guess:

.. math::
   \delta \mathcal{C}_{\ell}=\frac{1}{2} \sum_{\ell^\prime} \mathcal{F}_{\ell \ell^{\prime}}^{-1} \operatorname{Tr}\left[\left(\mathbf{C}^{-1} \frac{\partial \mathbf{S}}{\partial \mathcal{C}_{\ell}} \mathbf{C}^{-1}\right)\left(\mathbf{d} \mathbf{d}^{T}-\mathbf{C}\right)\right]
   :label: cell

.. math::
   \mathcal{F}_{\ell \ell^{\prime}}=\frac{1}{2} \operatorname{Tr}\left[\mathbf{C}^{-1} \frac{\partial \mathbf{S}}{\partial \mathcal{C}_{\ell}} \mathbf{C}^{-1} \frac{\partial \mathbf{S}}{\partial \mathcal{C}_{\ell^{\prime}}} \right]
   :label: fisher_ell

where I've left out all the math to get the first and second derivatives. **Note: I will use :math:`\mathcal{C}` for bandpowers and :math:`C` for covariance. Similarly, the Fisher matrix will be :math:`\mathcal{F}` and the transfer function will be :math:`F`.**

Now, instead of iterating on the steps toward the maximum, XFaster iterates towards the bandpowers themselves. It does this by reconfiguring the second term in the trace in Equation :math:numref:`cell`, which should iteratively get closer to zero, and instead reformats it to be the estimate of the measured signal:

.. math::
   \mathcal{C}_{\ell}=\frac{1}{2} \sum_{\ell'} \mathcal{F}_{\ell \ell^{\prime}}^{-1} \operatorname{Tr}\left[\left(\mathbf{C_{\ell'}}^{-1} \frac{\partial \mathbf{S_{\ell'}}}{\partial \mathcal{C}_{\ell'}} \mathbf{C_{\ell'}}^{-1}\right)\left(\mathbf{C}_{\ell'}^{o b s}-\langle\mathbf{N_{\ell'}}\rangle\right)\right]
   :label: eq12

where the :math:`\langle\mathbf{N}\rangle` is the ensemble average of the noise simulations, needed to debias the total covariance of the data to leave an estimate of signal alone.

From here, XFaster makes a few more approximations to make the matrix operations manageable. We approximate our noise to be diagonal and uncorrelated with signal, and the signal will be averaged into bins to reduce correlations among modes from using less than the full sky. So now, the covariance for the cut sky is approximated as:

.. math::
   \tilde{C}_{\ell m, \ell^{\prime} m^{\prime}}=\delta_{\ell \ell^{\prime}} \delta_{m m^{\prime}}\left(\tilde{\mathcal{C}}_{\ell}+\left\langle\tilde{N}_{\ell}\right\rangle\right)
   :label: eq13

The thing that our instrument measures is this pseudo-:math:`\tilde{\mathcal{C}}_\ell` spectrum. We ultimately want to know the full sky power spectrum, :math:`\mathcal{C}_\ell`. For TT, for example, that's related to our measured :math:`\tilde{\mathcal{C}}_\ell` s by

.. math::
   \tilde{\mathcal{C}}_{\ell}^{TT}=\sum_{\ell^{\prime}} K_{\ell \ell^{\prime}}^{TT} F_{\ell^{\prime}}^{TT} B_{\ell^{\prime}}^{2} \mathcal{C}_{\ell^{\prime}}^{TT}
   :label: eq14

where  :math:`K_{\ell, \ell'}` is the coupling kernel that accounts for mode mixing due to the non-orthogonality of the spherical harmonic basis on the cut sky, :math:`F_\ell` is the filter transfer function, and :math:`B_\ell` is the beam window function.

This is written on an :math:`\ell` by :math:`\ell` basis, but in practice we'll want to bin to reduce signal correlations and increase signal to noise, so we add the binning operator :math:`\chi_b`:

.. math::
   \tilde{\mathcal{C}}_{\ell}^{TT}=\sum_b q_b \sum_{\ell^{\prime}} K_{\ell \ell^{\prime}}^{TT} F_{\ell^{\prime}}^{TT} B_{\ell^{\prime}}^{2} \mathcal{C}_{\ell^{\prime}}^{TT} \chi_{b}\left(\ell^{\prime}\right)
   :label: eq15

where I've now added in a coefficient, :math:`q_b`, which accounts for any deviation of our measured bandpowers from the signal we expect our instrument to have measured. In practice, :math:`q_b` is actually what XFaster solves for. So now, instead of using :math:`\mathcal{C}_\ell` as the parameter we are optimizing, we instead solve for the maximum likelihood with respect to the bandpower deviations, :math:`q_b`:

.. math::
   q_{b}=\frac{1}{2} \sum_{b^{\prime}} \mathcal{F}_{b b^{\prime}}^{-1} \sum_{\ell} (2 \ell+1) \operatorname{Tr}\left[ \left(\tilde{\mathbf{D}}_{\ell}^{-1} \frac{\partial \tilde{\mathbf{S}}_{\ell}}{\partial q_{b^{\prime}}} \tilde{\mathbf{D}}_{\ell}^{-1}\right)\mathbf{g}\left(\tilde{\mathbf{D}}_{\ell}^{o b s}-\tilde{\mathbf{N}}_{\ell}\right)\mathbf{g}^T\right]
   :label: qb

.. math::
   \mathcal{F}_{b b^{\prime}}=\frac{1}{2} \sum_{\ell} (2 \ell+1)\operatorname{Tr}\left[\tilde{\mathbf{D}}_{\ell}^{-1} \frac{\partial \tilde{\mathbf{S}}_{\ell}}{\partial q_{b}} \tilde{\mathbf{D}}_{\ell}^{-1} \mathbf{g}\frac{\partial \tilde{\mathbf{S}}_{\ell}}{\partial q_{b^{\prime}}}\mathbf{g}^T\right]
   :label: fisher

where now instead of solving for just TT for one map, I'm generalizing to a matrix form where

.. math::
   \tilde{\mathbf{D}}_{\ell}=
   \begin{bmatrix}
   \tilde{\mathbf{D}}_{\ell}^{1x1} & \tilde{\mathbf{D}}_{\ell}^{1x2} & \tilde{\mathbf{D}}_{\ell}^{1x3} & \cdots & \tilde{\mathbf{D}}_{\ell}^{1xN} \\
   \tilde{\mathbf{D}}_{\ell}^{2x1} & \tilde{\mathbf{D}}_{\ell}^{2x2} & \tilde{\mathbf{D}}_{\ell}^{2x3} & \cdots & \vdots \\
   \tilde{\mathbf{D}}_{\ell}^{3x1} & \tilde{\mathbf{D}}_{\ell}^{3x2} & \tilde{\mathbf{D}}_{\ell}^{3x3} & \cdots & \vdots \\
   \vdots & \vdots & \vdots & \ddots & \vdots \\
   \tilde{\mathbf{D}}_{\ell}^{Nx1} & \cdots & \cdots & \cdots & \tilde{\mathbf{D}}_{\ell}^{NxN}\\
   \end{bmatrix}
   :label: dell

where :math:`N` is the number of maps, and each element of the above matrix is a 3x3 subblock of :math:`\tilde{C}_\ell` s for that map cross (*note: this the the full covariance, :math:`\tilde{C}_\ell` , not only the signal part, :math:`\tilde{\mathcal{C}}_\ell` *):

.. math::
   \tilde{\mathbf{D}}_{\ell}^{1\times 1}=\left[\begin{array}{ccc}{\tilde{\mathrm{C}}_{\ell}^{T T}} & {\tilde{\mathrm{C}}_{\ell}^{T E}} & {\tilde{\mathrm{C}}_{\ell}^{T B}} \\ {\tilde{\mathrm{C}}_{\ell}^{T E}} & {\tilde{\mathrm{C}}_{\ell}^{E E}} & {\tilde{\mathrm{C}}_{\ell}^{E B}} \\ {\tilde{\mathrm{C}}_{\ell}^{T B}} & {\tilde{\mathrm{C}}_{\ell}^{E B}} & {\tilde{\mathrm{C}}_{\ell}^{B B}}\end{array}\right]_{1\times 1}
   :label: eq19

We've also reduced the trace over $\ell$ in equations :math:numref:`cell` and :math:numref:`fisher_ell` to the number of modes we measure, assuming isotropy: :math:`\sum_{\ell}(2\ell+1)\mathbf{gg}^T`, where :math:`g` is a weighting factor accounting for the effective number of degrees of freedom of the map.  And the trace in equations :math:numref:`qb` and :math:numref:`fisher` is over the various map cross spectrum components.

There is some complication that arises from building the non-TT components of the signal covariance, which is that there is mixing between T :math:`\leftrightarrow` E,B and E :math:`\leftrightarrow` B caused by the masking. We account for this with the proper combination of shape operators, :math:`\tilde{\mathcal{C}}_{b\ell}`, along with their associated amplitudes, where the shape operators are defined below:

.. math::
   \begin{aligned}
   \tilde{\mathcal{C}}_{b \ell}^{T T}&=\sum_{\ell^{\prime}} K_{\ell \ell^{\prime}} F_{\ell^{\prime}}^{T T} B_{\ell^{\prime}}^{2} \mathcal{C}_{\ell^{\prime}}^{TT} \chi_{b}\left(\ell^{\prime}\right) \\
   {}_\pm \tilde{\mathcal{C}}_{b \ell}^{EE}&=\sum_{\ell^{\prime}} {}_\pm K_{\ell \ell^{\prime}} F_{\ell^{\prime}}^{EE} B_{\ell^{\prime}}^{2} \mathcal{C}_{\ell^{\prime}}^{EE} \chi_{b}\left(\ell^{\prime}\right) \\
   {}_\pm \tilde{\mathcal{C}}_{b \ell}^{BB}&=\sum_{\ell^{\prime}} {}_\pm K_{\ell \ell^{\prime}} F_{\ell^{\prime}}^{BB} B_{\ell^{\prime}}^{2} \mathcal{C}_{\ell^{\prime}}^{BB} \chi_{b}\left(\ell^{\prime}\right) \\
   \tilde{\mathcal{C}}_{b \ell}^{TE}&=\sum_{\ell^{\prime}} {}_\times K_{\ell \ell^{\prime}} F_{\ell^{\prime}}^{TE} B_{\ell^{\prime}}^{2} \mathcal{C}_{\ell^{\prime}}^{TE} \chi_{b}\left(\ell^{\prime}\right) \\
   \tilde{\mathcal{C}}_{b \ell}^{TB}&=\sum_{\ell^{\prime}} {}_\times K_{\ell \ell^{\prime}} F_{\ell^{\prime}}^{TB} B_{\ell^{\prime}}^{2} \mathcal{C}_{\ell^{\prime}}^{TB} \chi_{b}\left(\ell^{\prime}\right) \\
   \tilde{\mathcal{C}}_{b \ell}^{EB}&=\sum_{\ell^{\prime}} ({}_+ K_{\ell \ell^{\prime}}-{}_- K_{\ell \ell^{\prime}}) F_{\ell^{\prime}}^{EB} B_{\ell^{\prime}}^{2} \mathcal{C}_{\ell^{\prime}}^{EB} \chi_{b}\left(\ell^{\prime}\right) \\
   \end{aligned}
   :label: cbl

The shape operators, or "Cee-bee-ells" are simply understood to be the binned power we would expect given what we know of the coupling between our experiment and the sky. We have different shape expectations for the different signals we measure. Chiefly, the four that XFaster has currently implemented are CMB, dust, residual noise (that is, noise that is not accounted for in the noise simulation ensemble), and null signal. Each of these modifies the equation above somewhat, and we'll go into more detail about that further on in this guide.

The signal component of the covariance can then be written as

.. math::
   \tilde{\mathbf{S}}_\ell=
   \begin{bmatrix}
   \sum_b q_b^{TT}\tilde{\mathcal{C}}_{b\ell}^{TT} & \sum_b q_b^{TE}\tilde{\mathcal{C}}_{b\ell}^{TE} & \sum_b q_b^{TB}\tilde{\mathcal{C}}_{b\ell}^{TB} \\
   -- & \sum_b q_b^{EE} {}_+\tilde{\mathcal{C}}_{b\ell}^{EE}+ \sum_b q_b^{BB} {}_-\tilde{\mathcal{C}}_{b\ell}^{BB} & \sum_b q_b^{EB}\tilde{\mathcal{C}}_{b\ell}^{EB} \\
   -- & -- & \sum_b q_b^{BB} {}_+\tilde{\mathcal{C}}_{b\ell}^{BB}+ \sum_b q_b^{EE} {}_-\tilde{\mathcal{C}}_{b\ell}^{EE} \\
   \end{bmatrix}
   :label: signal

To construct equations :math:numref:`qb` and :math:numref:`fisher`, we need to take the derivatives of equation :math:numref:`signal` with respect to each :math:`q_b`. It's straightforward to read off the derivative terms:

.. math::
   \begin{align}
   \frac{\partial \tilde{\mathbf{S}}_{\ell}}{\partial q_{b}^{TT}} &=
   \begin{bmatrix}
   \tilde{\mathcal{C}}_{b\ell}^{TT} & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \\
   \end{bmatrix}
   \nonumber
   &
   \frac{\partial \tilde{\mathbf{S}}_{\ell}}{\partial q_{b}^{TE}} &=
   \begin{bmatrix}
   0 & \tilde{\mathcal{C}}_{b\ell}^{TE} & 0 \\ \tilde{\mathcal{C}}_{b\ell}^{TE} & 0 & 0 \\ 0 & 0 & 0 \\
   \end{bmatrix}
   \nonumber
   \\
   \nonumber
   \\
   \frac{\partial \tilde{\mathbf{S}}_{\ell}}{\partial q_{b}^{EE}} &=
   \begin{bmatrix}
   0 & 0 & 0 \\ 0 & {}_+\tilde{\mathcal{C}}_{b\ell}^{EE} & 0 \\ 0 & 0 & {}_-\tilde{\mathcal{C}}_{b\ell}^{EE} \\
   \end{bmatrix}
   \nonumber
   &
   \frac{\partial \tilde{\mathbf{S}}_{\ell}}{\partial q_{b}^{BB}} &=
   \begin{bmatrix}
   0 & 0 & 0 \\ 0 & {}_-\tilde{\mathcal{C}}_{b\ell}^{BB} & 0 \\ 0 & 0 & {}_+\tilde{\mathcal{C}}_{b\ell}^{BB} \\
   \end{bmatrix}
   \\
   \nonumber
   \\
   \frac{\partial \tilde{\mathbf{S}}_{\ell}}{\partial q_{b}^{TB}} &=
   \begin{bmatrix}
   0 & 0 & \tilde{\mathcal{C}}_{b\ell}^{TB} \\ 0 & 0 & 0 \\ \tilde{\mathcal{C}}_{b\ell}^{TB} & 0 & 0 \\
   \end{bmatrix}
   \nonumber
   &
   \frac{\partial \tilde{\mathbf{S}}_{\ell}}{\partial q_{b}^{EB}} &=
   \begin{bmatrix}
   0 & 0 & 0 \\ 0 & 0 & \tilde{\mathcal{C}}_{b\ell}^{EB} \\ 0 & \tilde{\mathcal{C}}_{b\ell}^{EB} & 0 \\
   \end{bmatrix}
   \nonumber
   \\
   \end{align}
   :label: dsdqb

So now everything is set up that we need, and we just need to build the ingredients. The rest of this document will be how we get each of the terms, but the main engine of XFaster, once it has all the ingredients, is to iterate on equations :math:numref:`qb` and :math:numref:`fisher`. So,

 1. Start with an initial guess at the $q_b$s, which we set to be 1.
 2. Compute the Fisher matrix with Equation :math:numref:`fisher`.
 3. Plug that into Equation :math:numref:`qb` to get a new guess for :math:`q_b`.
 4. Repeat until some convergence criterion is met.

We can use all these same tools to also fit for the transfer function-- instead of using :math:`\tilde{\mathbf{D}}_\ell^{obs}-\tilde{\mathbf{N}}_\ell` for our measured signal spectrum, we just use the ensemble average of the signal simulations, and set :math:`F_\ell` in Equation :math:numref:`cbl` to be 1. Then, the :math:`q_b` s that pop out are just the transfer function itelf, and the inverse Fisher matrix gives the error on the transfer function.
