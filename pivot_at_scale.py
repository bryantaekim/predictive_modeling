'''
Perform better than the pivot at scale
'''
cols = [columns]
target = ''
query1 = "select m.col1, m.col2, m.col3, m."+target
for c in cols:
    foo = ", collect_list("+c+"[0]) as "+c
    query1 += foo

query2 = query1 + " from (select col1, col2, col3, "+target
for c in cols:
    foo = ", array("+c+") as "+c
    query2 += foo

query = query2 + " from df order by col1, col5 desc) m group by 1,2,3,4"
