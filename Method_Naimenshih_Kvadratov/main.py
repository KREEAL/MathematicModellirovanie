import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sample_x_array = [1, 2, 3, 4, 5, 6]
sample_y_array = [1, 1.5, 3, 4.5, 7, 8.5]
# 13 вариант


# в задании второе число - 16, но потом идет 13, я принял решение взять 6
#sample_x_array = [1, 6, 13, 46, 61, 76]
#sample_y_array = [0.5, 4.0, 6.9, 8.8, 10.9, 12.1]

def opredelitel_2_poryadka(a11, a12, a21, a22):
    return a11 * a22 - a12 * a21

def opredelitel_3_poryadka(a11, a12,a13, a21, a22,a23,a31,a32,a33):
    return a11*a22*a33+a21*a13*a32+a31*a12*a23-a31*a22*a13-a21*a12*a33-a11*a32*a23

#y=ax+b
# для линейной работает шикарно
def find_a_and_b_linear(list_x: list, list_y: list):
    a1 = sum(list(map(lambda x: x ** 2, list_x)))
    b1 = sum(list_x)
    c1 = sum([x * y for x, y in zip(list_x, list_y)])
    a2 = sum(list_x)
    b2 = len(list_x)
    c2 = sum(list_y)

    delta = opredelitel_2_poryadka(a1, b1, a2, b2)
    delta1 = opredelitel_2_poryadka(c1, b1, c2, b2)
    delta2 = opredelitel_2_poryadka(a1, c1, a2, c2)
    koef_a = delta1 / delta
    koef_b = delta2 / delta
    return (koef_a, koef_b)

#y=b*x^a
def find_a_and_b_exponential(list_x: list, list_y: list):
    (alfa, beta) = find_a_and_b_linear([np.log(x) for x in list_x], [np.log(x) for x in list_y])
    b = np.exp(beta)
    return (alfa, b)

#y=b*e^(a*x)
def find_a_and_b_pokazateln(list_x: list, list_y: list):
    (alfa, beta) = find_a_and_b_linear(list_x, [np.log(x) for x in list_y])
    b = np.exp(beta)
    return (alfa, b)

#y=ax^2+bx+c
def find_a_and_b_kvadratich(list_x: list, list_y: list):
    a1 = sum(list(map(lambda x: x ** 4, list_x)))
    a2 = sum(list(map(lambda x: x ** 3, list_x)))
    a3 = sum(list(map(lambda x: x ** 2, list_x)))
    b1 = a2
    b2 = a3
    b3 = sum(list_x)
    c1 = a3
    c2 = b3
    c3 = len(list_x)
    d1 = sum([(x**2) * y for x, y in zip(list_x, list_y)])
    d2 = sum([x * y for x, y in zip(list_x, list_y)])
    d3 = sum(list_y)
    delta=opredelitel_3_poryadka(a1,b1,c1,a2,b2,c2,a3,b3,c3)
    delta1=opredelitel_3_poryadka(d1,b1,c1,d2,b2,c2,d3,b3,c3)
    delta2=opredelitel_3_poryadka(a1,d1,c1,a2,d2,c2,a3,d3,c3)
    delta3=opredelitel_3_poryadka(a1,b1,d1,a2,b2,d2,a3,b3,d3)

    return(delta1/delta,delta2/delta,delta3/delta)

def get_error_quantity(whitelist,list_y):
    return sum([(x - y)**2 for x, y in zip(whitelist, list_y)])



print(find_a_and_b_linear(sample_x_array, sample_y_array))
print(find_a_and_b_exponential(sample_x_array, sample_y_array))
print(find_a_and_b_pokazateln(sample_x_array, sample_y_array))
print(find_a_and_b_kvadratich(sample_x_array,sample_y_array))




linear_koefs = find_a_and_b_linear(sample_x_array,sample_y_array)
y_linear = [x*linear_koefs[0]+linear_koefs[1] for x in sample_x_array]

exponential_koefs = find_a_and_b_exponential(sample_x_array,sample_y_array)
y_exponential = [exponential_koefs[1]*x**exponential_koefs[0] for x in sample_x_array]

pokazateln_koefs = find_a_and_b_pokazateln(sample_x_array,sample_y_array)
y_pokazateln = [pokazateln_koefs[1]*np.exp(pokazateln_koefs[0]*x) for x in sample_x_array]

kvadratich_koefs = find_a_and_b_kvadratich(sample_x_array,sample_y_array)
y_kvadratich = [kvadratich_koefs[0]*(x**2)+kvadratich_koefs[1]*x+kvadratich_koefs[2] for x in sample_x_array]
yses = {"Линейный":{sum([(x - y)**2 for x, y in zip(sample_y_array, y_linear)]):linear_koefs},
        "Экспоненциаьный":{sum([(x - y)**2 for x, y in zip(sample_y_array, y_exponential)]):exponential_koefs},
        "Показательный":{sum([(x - y)**2 for x, y in zip(sample_y_array, y_pokazateln)]):pokazateln_koefs},
        "Квадратичный":{sum([(x - y)**2 for x, y in zip(sample_y_array, y_kvadratich)]):kvadratich_koefs}}







errs_list = []
for item in yses:
    for key in yses[item]:
        errs_list.append((item,key))
print(errs_list)
print()
min_el = errs_list[0]
for element in errs_list:
    if element[1]<min_el[1]:
        min_el = element
print(min_el)

plt.plot(sample_x_array,y_linear,c='red')
plt.plot(sample_x_array,y_exponential,c='green')
plt.plot(sample_x_array,y_pokazateln,c='black')
plt.plot(sample_x_array,y_kvadratich,c='orange')
plt.scatter(sample_x_array,sample_y_array)
plt.show()