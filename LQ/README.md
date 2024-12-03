# code for the LQG control problem.

## Folder Structure

The repository is organized as follows:

* **Adam_CBO_case1/**: Contains code for Adam CBO method for terminal cost $g(\mathbf x) = \ln \frac{1 + \| \mathbf x \|^2}{2}$ in $2,4,8,16$ dimensional probelms.

* **Adam_CBO_case2/**: Contains code for Adam CBO method for terminal cost $g(\mathbf x) = \ln \frac{1 + (\| \mathbf x \|^2 - 1)^2}{2}$ in $2,4,8,16$ dimensional probelms .

* **Adam_CBO_case5/**: Contains code for Adam CBO method for terminal cost $g(\mathbf x) = \ln \frac{1 + (\| \mathbf x \|^2 - 5)^2}{2}$ in $4$ dimensional probelms with $64,128,256,512,1024,2048$ SDE samplers to compute the expectation of the cost function.

* **BSDE/**: BSDE method for terminal cost $g(\mathbf x) = \ln \frac{1 + (\| \mathbf x \|^2 - 1)^2}{2}$ and  $g(\mathbf x) = \ln \frac{1 + (\| \mathbf x \|^2 - 5)^2}{2}$  in $2,4,8,16$ dimensional probelms  .

* **BSDE_case5_batch_size/**: BSDE method for terminal cost $g(\mathbf x) = \ln \frac{1 + (\| \mathbf x \|^2 - 5)^2}{2}$ in $4$ dimensional probelms with $64,128,256,512,1024,2048$ SDE samplers to compute the expectation of the cost function.


* **M_CBO_case1/**: Contains code for M-CBO method for terminal cost $g(\mathbf{x})=\ln\frac{1+\|\mathbf{x}\|^2}{2}$ in $2,4,8,16$ dimensional probelms.


* **value_function_case1/**: Contains code to compute the value function by take exception of the controled SDE for terminal cost $g(\mathbf{x})=\ln\frac{1+\|\mathbf{x}\|^2}{2}$  in $2$ dimensional probelms.

* **value_function_case2/**: Contains code to compute the value function by take exception of the controled SDE for terminal cost $g(\mathbf{x})=\ln\frac{1+(\|\mathbf{x}\|^2- 1)^2}{2}$ in $2$ dimensional probelms.

