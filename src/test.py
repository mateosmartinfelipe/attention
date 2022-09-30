def a():
    d = []
    helping(d)
    return d


def helping(d):
    d.append(2)


print(a())

"""
# Write your MySQL query statement  185. Department Top Three Salaries
WITH B AS (
    SELECT 
        ROW_NUMBER() OVER(PARTITION BY a.departmentId ORDER BY a.salary Desc) AS rank,
        a.salary ,a.departmentId 
    FROM (SELECT distinct salary ,departmentId FROM Employee) as a
    )

    SELECT
        Department.name as Department,
        Employee.name as Employee,
        B.salary as Salary
    FROM B
    JOIN Employee ON B.salary = Employee.salary and B.departmentId=Employee.departmentId
    JOIN Department ON B.departmentId = Department.id
    WHERE B.rank <=3
 
    

"""
