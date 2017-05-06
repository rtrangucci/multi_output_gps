Contains code for multiple output/Kronecker GPs in Stan. 

The following models are included in the repo:

- 1-dimensional GP with 3-dimensonal multivariate normal observations
- 2-dimensional Kronecker GP with Gaussian observations
- 2-dimensional Kronecker GP with Poisson observations
- 3-dimensional Kronecker GP with Gaussian observations
- 3-dimensional Kronecker GP with Poisson observations

Each .stan file has a matching R file that will generate fake data
corresponding the generative model in the Stan program. 

MIT License

Copyright (c) [2017] [Robert Trangucci]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
