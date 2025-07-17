# A
#work #algorithm
**Algorithm for a computing the dependency graph in Hive QL script (in a nutshell)**
```
for each table/view/insert in script:
	collect all the from clauses and join clauses
	generate the table to table ref adjacency list (map)
```
## AB
- Implement a collect function on the tree I can't find one.
* For the rascal initial code draft [[Dependency graph first attempt]]
* I did a second attempt on the algorithm this variant takes into account the effect of the use statement in a query script [[Dependency graph second attempt]]
* I tried to do a computation of the dependency of a bunch of files [[Dependency graph on scripts in a directory]]

---
### AAA
* The core of this algorithm if figuring out the right data structure the captures the as much information as possible [[Dependency graph data structure]]
* For the sake of illustration I want to visualize the dependency graph [[Visualize dependency graph]]
* I want to check if a graph has a cycle [[Find Cycle in the dependency graph]]