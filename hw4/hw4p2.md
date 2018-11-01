% hw4p2
% Zachary Moon
% 24-Oct-18


**The paper:**

|        |                                           |
| ------:|:----------------------------------------- |
| title  | A Rotated Principal Component Analysis 
|        | of the Interannual Variability of the 
|        | Northern Hemisphere 500 mb Height Field
| author | Horel, JD
| year   | 1981
| doi    | [10.1175/1520-0493(1981)109%3C2080:ARPCAO%3E2.0.CO;2](https://doi.org/10.1175/1520-0493(1981)109%3C2080:ARPCAO%3E2.0.CO;2)


## Rotated principal component analysis (PCA)

### What is rotated PCA?

Rotating of principal components (PCs) is the process of applying a linear transformation to a matrix of selected PCs. Rotated PCA is the analysis of the rotated PCs and corresponding empirical orthogonal functions (EOFs) or loading vectors. 

The purpose of this rotation is to reduce the geometrical impact of the orthogonality constraint on patterns identified by PCA. Such rotations tend to make the non-primary PCs easier to interpret. Some rotations keep the orthogonality of PCs (rigid rotations; orthogonal solutions) while some give partially correlated PCs (called oblique solutions). 


### How is it conducted?

In this study rotation of selected PCs is conducted using the **varimax** method. This method maximizes the *variance* of the $R^2$'s between each rotated PC and the input data time series ($R$ is the Pearson correlation coefficient). Note that this $R^2$ is equivalent to the fraction of variance explained by the PC, and to the variance of the PC, and to the corresponding covariance matrix eigenvalue for that PC. 

The normal PC solution, in contrast, maximizes the *sum* of those $r^2$'s. Rotated PCA generally gives a wider distribution of loadings (few large, many small) within each loading or eigen-vector, making the PCs easier to interpret. 

PCs that explain more than one unit of total normalized variance in the original data were selected for the rotation ("Guttman criterion"), i.e., data covariance matrix eigenvalue $> 1$. In the situations of the paper, usually about 20% of the PCs meet this criterion. Data reconstructed from the rotated PCs should match that of the selected initial PCs, i.e., the total variability explained will be the same (for the selected group). 


### Why is it useful?

Rotated PCA

* gives patterns ("loadings") that are easier to explain in a meteorological context
  * can more closely resemble observed anomaly fields
* gives results that are not as dependent on the specifics of chosen input data spatial domain in your area of interest


### What are the drawbacks?

Some of the drawbacks of using standard PCA for geophysical large datasets still apply.

* rotated PCs (which some say should just be called RCs (rotated components)) may still not really represent any physical/dynamical entity in the real data, such as an anomaly field of some meteorological variable
* the rotated PC solution presented here is just one of many-- they are many other rotations that can be done and much argument about which are the best in which situations

