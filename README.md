# machine-learning-2-week-7-km2-solved
**TO GET THIS SOLUTION VISIT:** [Machine Learning 2 Week 7-KM2 Solved](https://www.ankitcodinghub.com/product/machine-learning-2-week-7-km2-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;98837&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;Machine Learning 2 Week 7-KM2 Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column"></div>
<div class="column">
&nbsp;

Exercise Sheet 7

</div>
</div>
<div class="layoutArea">
<div class="column">
Exercise 1: Structured Prediction for Classification (20 P)

While structured output learning is typically used for predicting complex output signals such as sequences or trees, the same framework can also be used to address the more standard problem of classification. Let x, x′ ∈ Rd be two data points and y, y′ ∈ {1, . . . , C} their respective classes. Consider the structured output kernel

kstruct((x, y), (x′, y′)) = k(x, x′) · I(y = y′),

where k(x,x′) is a positive semi-definite kernel with associated feature map φ(x), and where I(·) is an

indicator function that is 1 when the argument is true and 0 otherwise.

(a) Show that the kernel kstruct((x, y), (x′, y′)) is positive semi-definite, that is, show that

NN

􏰄 􏰄 cicj kstruct((xi, yi), (xj , yj )) ≥ 0

i=1 j=1

for all input/output pairs (x1,y1),…,(xN,yN) and choice of real numbers c1,…,cN.

(b) Find a feature map φstruct(x,y) associated this kernel, i.e. satisfying

􏰕φstruct(x, y), φstruct(x′, y′)􏰖 = kstruct((x, y), (x′, y′))

for all pairs (x, y) and (x′, y′).

Exercise 2: Dual Formulation of Structured Output Learning (20 P)

In structured output learning, we look for a linear model in joint feature space that produces a large margin between the correct prediction and all other possible predictions. The primal formulation of this problem can be expressed as:

</div>
</div>
<div class="layoutArea">
<div class="column">
subject to the constraints:

∀Nn=1∀y̸=yn : w⊤Ψn,y ≥ 1 − ξn and ∀Nn=1 : ξn ≥ 0

where we have used the shortcut notation Ψn,y = φ(x, yn) − φ(x, y). (a) Show that the associated dual optimization problem is given by:

</div>
</div>
<div class="layoutArea">
<div class="column">
1 2 􏰄N min ∥w∥+C ξn

</div>
</div>
<div class="layoutArea">
<div class="column">
w,ξ 2 n=1

</div>
</div>
<div class="layoutArea">
<div class="column">
N1

max 􏰄 􏰄 αn,y − 􏰄 􏰄 αn,yαn′,y′ ⟨Ψn,y, Ψn′,y′ ⟩

</div>
</div>
<div class="layoutArea">
<div class="column">
α n=1 y̸=y

</div>
<div class="column">
2 n,n′ y̸=y nn

y′̸=yn′

</div>
</div>
<div class="layoutArea">
<div class="column">
subject to the constraints:

∀Nn=1∀y̸=yn : 0 ≤ αn,y and ∀Nn=1 :

</div>
<div class="column">
􏰄 αn,y ≤ C y̸=yn

</div>
</div>
<div class="layoutArea">
<div class="column">
(b) Assuming k((x,y),(x′,y′)) is the kernel that induces the feature map φ(x,y), express the dot product ⟨Ψn,y, Ψn′,y′ ⟩ in terms of this kernel.

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Exercise 3: Prediction of Output Sequences (20 P)

Consider output sequences to predict to be of the type y ∈ {−1, 1}L, and the feature map: φ(x,y)=􏰇x⊙y , 2·(y1…L−1 ⊙y2…L)􏰈

where ⊙ denotes the element-wise product between two vectors. The structured output model looks for the output y that maximizes the matching function, i.e.

max w⊤φ(x,y) y

with w ∈ R2L−1. In the following, we assume that L = 3, that the current parameter is w = (1, 1, 1, 1, 1) and that we receive the input x = (1, −1, 1).

(a) Show that the problem of maximizing the matching function simplifies to:

max 􏰗y + max 􏰗2yy −y + max 􏰗2yy +y􏰘􏰘􏰘

</div>
</div>
<div class="layoutArea">
<div class="column">
1 122 233 y1 ∈{−1,1} y2 ∈{−1,1} y3 ∈{−1,1}

</div>
</div>
<div class="layoutArea">
<div class="column">
(b) Find using the Viterbi procedure the best output (y1,y2,y3), that is, solve maxy3{} for every y2, then solve maxy2 {} for every y1, and then solve maxy1 {}. While doing so, keep track of the values in the sequence that have produced the respective maximums so that the optimal sequence can be reconstructed.

Exercise 4: Programming (40 P)

Download the programming files on ISIS and follow the instructions.

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="section">
<div class="section">
<div class="layoutArea">
<div class="column">
Exercise sheet 7 (programming) [SoSe 2021] Machine Learning 2

</div>
</div>
<div class="layoutArea">
<div class="column">
Structured Output Learning

In this exercise, we consider a data completion task to be solved with structured output learning. The dataset is based on the dataset of the previous programming sheet on splice sites classification. We would like to be able to reconstruct a nucleotide sequence when one of the nucleotides is missing. One such incomplete sequence of nucleotides is shown in the image below

where the question mark indicates the missing nucleotide. We would like to make use of the degree kernel that was used in the previous programming sheet. It was shown to represent genes data efficiently near the splice sites. For our completion task, we adopt a structured output learning approach, where the candidate value for replacing the missing nucleotide is also part of the kernel feature map. Interestingly, with this approach, the kernel can still apply to the same type of input data (i.e. continuous gene sequences) as in the standard splice classification setting.

The structured output problem is defined as solving the soft-margin SVM optimization problem:

</div>
</div>
<div class="layoutArea">
<div class="column">
where for all inputs pairs (xi , yi )N i=1

constraints hold:

</div>
<div class="column">
min ∥w∥2 + C∑N ξi w,b i=1

representing the genes sequences and the true value of the missing nucleotide, the following

</div>
</div>
<div class="layoutArea">
<div class="column">
w⊤ φ(xi , yi ) + b ≥ 1 − ξi ∀zi ∈{A,T,C,G}∖yi : w⊤φ(xi,zi)+b≤−1+ξi

</div>
</div>
<div class="layoutArea">
<div class="column">
ξi ≥ 0 Once the SVM is optimized, a missing nucleotide y for sequence x can be predicted as:

y(x) = arg max w⊤ φ(x, z). z∈{A,T,C,G}

</div>
</div>
<div class="layoutArea">
<div class="column">
The feature map φ(x, z) is implicitely defined by the degree kernel between gene sequences r and r′ given as kd(r,r′) = ∑L−d+11{r[i…i+d]=r′[i…i+d]}

</div>
</div>
<div class="layoutArea">
<div class="column">
i=1

where r is built as the incomplete genes sequence x with missing nucleotide “?” set to z, and where r[i … i + d] is a subsequence

</div>
</div>
<div class="layoutArea">
<div class="column">
of r starting at position i and of length d.

Loading the Data

The following code calls a function from the file utils.py that loads the data in the IPython notebook. Note that only the 20 nucleotides nearest to the splice site are returned. The code then prints the first four gene sequences from the dataset, where the character “?” denotes the missing nucleotide. The label associated to each incomplete genes sequences (i.e. the value of the missing nucleotide “?”) is shown on the right.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [1]:

</div>
<div class="column">
import utils

<pre>Xtrain,Xtest,Ytrain,Ytest = utils.loaddata()
</pre>
<pre>print("".join(Xtrain[0])+"  ?="+Ytrain[0])
print("".join(Xtrain[1])+"  ?="+Ytrain[1])
print("".join(Xtrain[2])+"  ?="+Ytrain[2])
print("".join(Xtrain[3])+"  ?="+Ytrain[3])
</pre>
<pre>CAACGATCCAT?CATCCACA  ?=C
CAGGACGGTCA?GAAGATCC  ?=G
AAAAAGATGA?GTGGTCAAC  ?=A
TGTCGGTTA?CAATGATTTT  ?=C
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
It can be observed from the output that the missing nucleotide is not always at the same position. This further confirms that the problem cannot be treated directly as a standard multiclass classification problem. Note that in this data, we have artificially removed nucleotides in the training and test set so that we have labels y available for training and evaluation.

</div>
</div>
</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="section">
<div class="layoutArea">
<div class="column">
Generating SVM Data (15 P)

In the SVM structured output formulation, the data points (xi , yi ) denote the true genes sequences and are the SVM positive examples. To be able to train the SVM, we need to generate all possible examples ((xi,zi))zi∈{A,T,C,G}.

Your first task is to implement a function builddata(X,Y) that receives as input the dataset of size (N x L) of incomplete gene sequences X where N is the number of gene sequences and L is the sequence length, and where Y of size N contains the values of missing nucleotides.

Your implementation should produce as output an extended dataset of size (4N x L) . Also, the function should return a vector of labels T of size 4N that is +1 for positive SVM examples and -1 for negative SVM examples. For repeatability, ensure that all

</div>
</div>
<div class="layoutArea">
<div class="column">
modifications of the same gene sequence occur in consecutive order in the outputs

In [2]: def builddata(X,Y):

<pre>                  # Replace by your own code
</pre>
<pre>                  import solutions
</pre>
<pre>                  XZ,T = solutions.builddata(X,Y)
</pre>
# —

assert(len(XZ)==len(T)==4*len(X)==4*len(Y)) return XZ,T

</div>
<div class="column">
XZ and

</div>
<div class="column">
T .

</div>
</div>
<div class="layoutArea">
<div class="column">
Your implementation can be tested by running the following code. It applies the function to the training and test sets and prints the first 12 examples in the training set (i.e. all four possible completions of the first three gene sequences).

</div>
</div>
<div class="layoutArea">
<div class="column">
In [3]:

</div>
<div class="column">
<pre>XZtrain,Ttrain = builddata(Xtrain,Ytrain)
XZtest,_       = builddata(Xtest ,Ytest )
</pre>
print(XZtrain)

for xztrain,ttrain in zip(XZtrain[:12],Ttrain[:12]):

print(“”.join(xztrain)+’ %+1d’%ttrain)

<pre>[['C' 'A' 'A' ... 'A' 'C' 'A']
 ['C' 'A' 'A' ... 'A' 'C' 'A']
 ['C' 'A' 'A' ... 'A' 'C' 'A']
 ...
</pre>
<pre> ['T' 'C' 'C' ... 'G' 'C' 'T']
 ['T' 'C' 'C' ... 'G' 'C' 'T']
 ['T' 'C' 'C' ... 'G' 'C' 'T']]
</pre>
<pre>CAACGATCCATACATCCACA -1
CAACGATCCATTCATCCACA -1
CAACGATCCATCCATCCACA +1
CAACGATCCATGCATCCACA -1
CAGGACGGTCAAGAAGATCC -1
CAGGACGGTCATGAAGATCC -1
CAGGACGGTCACGAAGATCC -1
CAGGACGGTCAGGAAGATCC +1
AAAAAGATGAAGTGGTCAAC +1
AAAAAGATGATGTGGTCAAC -1
AAAAAGATGACGTGGTCAAC -1
AAAAAGATGAGGTGGTCAAC -1
</pre>
</div>
</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="section">
<div class="layoutArea">
<div class="column">
SVM Optimization and Sequences Completion (25 P)

In this section, we would like to create a function that predicts the missing nucleotides in the gene sequences. The function should be structured as follows: First, we build the kernel training and test matrices using the function utils.getdegreekernels and using the specified degree parameter. Using scikit-learn SVM implementation ( sklearn.svm.SVC ) to train the SVM associated to the just computed kernel matrices and the target vector Ttrain . Use the default SVM hyperparameter C=1 for training.

After training the SVM, we would like to compute the predictions for the original structured output problem, that is, for each original gene sequence in the training and test set, the choice of missing nucleotide value for which the SVM prediction value is highest. The outputs Ptrain and Ptest denote such predictions and should be arrays of characters A,T,C,G of same size as the vectors of true nucleotides values Ytrain and Ytest .

(Hint: You should consider that in some cases there might be not exactly one missing nucleotide value that produces a positive SVM classification. In such cases, we would still like to find the unique best nucleotide value based on the value of the discriminant function for this particular data point. A special function of scikit-learn’s SVC class exists for that purpose.)

In [4]: def predict(XZtrain,XZtest,Ttrain,degree):

<pre>                  # Replace by your own code
</pre>
<pre>                  import solutions
</pre>
<pre>                  Ptrain,Ptest = solutions.predict(XZtrain,XZtest,Ttrain,degree)
</pre>
# —

return Ptrain,Ptest

The code below tests the prediction function above with different choices of degree parameters for the kernel. Note that running the code can take a while (up to 3 minutes) due to the relatively large size of the kernel matrices. If the computation time becomes problematic, consider a subset of the dataset for development and only use the full version of the dataset once you are ready to produce the final version of your notebook.

</div>
</div>
<div class="layoutArea">
<div class="column">
In [5]:

</div>
<div class="column">
import solutions

for degree in [1,2,3,4,5,6]:

<pre>    Ptrain,Ptest = predict(XZtrain,XZtest,Ttrain,degree)
</pre>
<pre>    acctr = (Ytrain==Ptrain).mean()
    acctt = (Ytest ==Ptest ).mean()
</pre>
print(‘degree: %d train accuracy: %.3f test accuracy: %.3f’%(degree,acctr,acct t))

<pre>degree: 1  train accuracy: 0.295  test accuracy: 0.281
degree: 2  train accuracy: 0.517  test accuracy: 0.530
degree: 3  train accuracy: 0.564  test accuracy: 0.516
degree: 4  train accuracy: 0.804  test accuracy: 0.499
degree: 5  train accuracy: 0.965  test accuracy: 0.492
degree: 6  train accuracy: 0.998  test accuracy: 0.487
</pre>
</div>
</div>
</div>
</div>
<div class="page" title="Page 6"></div>
<div class="page" title="Page 7"></div>
<div class="page" title="Page 8"></div>
