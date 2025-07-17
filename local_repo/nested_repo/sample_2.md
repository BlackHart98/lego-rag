#work #algorithm
**Algorithm for a computing the dependency graph in Hive QL script (in a nutshell)**
```
for each table/view/insert in script:
	collect all the from clauses and join clauses
	generate the table to table ref adjacency list (map)
```