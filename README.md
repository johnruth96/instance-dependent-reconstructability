# Experiments for Local Refinement of Semantics for Avoiding Anomalies in Graph Data Joins

This repository contains our preliminary experiments for our submitted paper *Local Refinement of 
Semantics for Avoiding Anomalies in Graph Data Joins*
to the ER 2022 conference.

## Getting Started

Please install the requirements in `requirements.txt`. The software is written in Python and should be executed with Python 3.9.x.
Additionally, you need the GDBMS *BlazeGraph* as JAR-file.


## Method

The high-level method is the following:
We compare our method against the two extremes of reification: Complete reification and no reification.
We generate relational data and decompose it into three KGs, one for each method:
One KG is a complete star decomposition (no reification), one KG uses our minimal reification approach, and one KG uses complete reification (no star decomposition).
On each graph, we perform four queries and measure the execution time.
Additionally, we measure the total time, including the rewriting time for our method.
Please note that the KG with complete star decomposition is a lossy decomposition.

### Dataset Generation
The relational data is a synthetically generated dataset.
It consists of a complex relation with $n>2$ attributes and a binary relation.
The generation uses the following parameters: (1) number of entities $n$, (2) ratio of reified entities $r$, (3) number tuples per entity $t$ in the complex relation and (4) number of attributes $m$.
We want to control the structure of the KG, which is a result of the decomposition of the two relations.
The later KG has $n$ entities, from which $r*n$ have reified facts and $(1-r)*n$ have regular facts.
%We generate $n$ entities, some of which are reified.
%This distinction is important because we want to control the number of entities that have reified facts in the later KG.
We generate $n$ entities in the complex relation.
For $r*n$ entities (reified facts), we generate $t$ tuples.
We generate a single tuple in the complex relation for $(1-r)*n$ entities (regular facts).
Since each entity in the KG should have the same amount of facts, we generate tuples in the binary relation for the $(1-r)*n$ entities.
%An entity that should not be reified is precisely one tuple in the complex relation and some for the binary relation (we explain the exact number later on).

In the experiments, we generate multiple datasets with different values for $r$ because we want to investigate the relationship between the ratio of reification in a KG and the execution time for our queries.
So, an important aspect is that the datasets for different values of $r$ are comparable.
If we increase the ratio of reification, we generate more tuples in the complex relation and, thus, more triples in the graph.
Therefore, the dataset for $r=1$ contains more information than the dataset for $r=0$, so the datasets are not comparable anymore.

We correct this issue by ensuring that the graph with star decomposition always has the same number of triples, i.e., the number of regular facts stays the same.
The graphs with minimal and complete reification contain more triples but still express the same information.
The additional triples are only essential to ensure the correct decomposition of the information.
In a dataset with $r=1$, each entity requires reification.
The corresponding graph in star decomposition contains $n*t*(m-1)$ triples (the first attribute is the subject).
Each dataset should map to $n*t*(m-1)$ triples in star decomposition (disregarding the parameter $r$).
For $r<1$, we replace one $m$-ary tuple with $m-1$ tuples in the binary relation.
For $r=0$, each entity appears once in the complex relation but more often in the binary relation.

Let $x$ be an entity that should be reified.
We generate $t$ $m$-ary tuples.
Let $y$ be an entity that should not be reified.
We generate a single $m$-ary tuple and $(t-1)*(m-1)$ binary tuples.
For each attribute, we chose a value from a uniform distribution over $n*t*(m-1)$ (almost unique values).

We generate 11 datasets, starting from $r=0$ to $r=1$ in steps of $0.1$.
Each dataset consists of $n=20000$ entities and $t=6$ tuples per entity.
Each dataset has $m=4$ attributes, so there are three additional attributes to the subject attribute.

### Knowledge Graphs
We transform a dataset into KGs for complete, minimal and no reification.
We want to compare the reification techniques to standard reification, $n$-ary relations and singleton property.
We generate a graph for each technique for each method (except no reification).
The no reification method only needs one graph (only one lossy decomposition).
This results in 7 KGs per dataset.

We query each graph with four SPARQL queries.
For each query, we only count the number of results.
Here, we use small queries as it is assumed that queries are generally small.
The graphs without reification are queried with a regular SPARQL query.
Depending on the technique, the graph with complete reification is queried with a reified version.
The graphs with minimal reification are queried with the rewritten query of our rewriting algorithm.
We measure the time for the rewriting and the query execution.
The time for query rewriting for the complete and no reification are zero.
Please note that the initialization of our query rewriter does *not* count into the rewriting time.
The initialization is the same as starting a database system, e.g., loading indexes, and is part not of the actual rewriting process.
Each query is executed seven times to calculate the average execution time.

The graphs are named graphs and are serialized in the N-Quads format.
We use the SPARQL engine BlazeGraph.
The graphs with complete and no reification act as a baseline, i.e., the two extremes of using reification in a graph are our baselines.
Our approach is implemented in Python.


## Results

![Avg. Execution time Q1](./paper/time_q1.png)

![Avg. Execution time Q2](./paper/time_q2.png)

Diagrams for the average execution times and the sizes are shown above.

The figures above show the average query execution times for $Q_1$ and $Q_2$ (aggregated over all methods).
The diagram shows four lines (three, respectively): one for no reification, one for complete reification and two for minimal reification.
For minimal reification, we show the total execution time (including the rewriting time) and the query's execution time only.
However, the rewriting time is so small that a difference is hardly visible.

The graphs show that the execution time increases with an increasing value of $r$.
The no reification method is always faster than the other two methods.
It is visible that the minimal approach executes faster than complete reification for queries $Q_1$ (and $Q_3$).
This superiority becomes smaller with an increasing value of $r$.

For queries $Q_2$ (and $Q_4$), the execution times of minimal reification are slightly higher than complete reification but still comparable.
The execution time for minimal and no reification increases faster than for no reification.

The query size is shown above.
The average query size for the no reification method is 2.25 triples.
For complete reification, it is 8.5 triples and 12.8 triples for minimal reification, which is about 1.5 times higher than for complete reification.