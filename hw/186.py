"""Author: Anthony Bajoua, preliminary grade calculator for cs186, enter
score you received on each assignment. I'm pretty sure it's all right
for any concerns comment on the piazza post and I'll fix!"""

vitamins = #CHANGE/10
proj1 = #CHANGE/100
proj2 = #CHANGE/100
proj3 = #CHANGE/100
proj4 = #CHANGE/100
proj5 = #CHANGE/100

mt1 = #CHANGE/84.0001
mt2 = #CHANGE/85
high, low = max(mt1, mt2), min(mt1, mt2)

#CHANGE IF YOU WANT TO ESTIMATE YOUR GRADE ON THE PAPER
paper = 0/100

#CHANGE TO TRUE IF YOU WILL SKIP THE PAPER
skip_paper = False

total = (.1 * vitamins) + (.07 * (proj1 + proj2 + proj3 + proj4)) + (.12 * proj5) \
            + (.12 * high) + (.08 * low) + (.15 * paper)

#CHANGE to True if we reach 80%. It's really short and the time you're saving here you could do it all your course evals (;
final_survey = False
if final_survey:
    total += .01


pnp = False #CHANGE to True if you will PNP the class.
#-------------------------------------NOTHING TO CHANGE BELOW ME--------------------------------------------#

bins = [('A+', .88), ('A', .77), ('A-', .74), ('B+', .71), ('B', .58), \
            ('B-', .55), ('C+', .52), ('C', .48), ('C-', .46), ('D+', .44), ("D", .42), ("D-", .4)]

for j in range(len(bins) - 1, -1, -1):
    if total >= bins[j][1]:
        grade = bins[j][0]
    else:
        break


print(f"Your current raw score is a {total} \nYour current grade would be a {grade} given any estimates made on proj5 and the paper (if any)\n")

if pnp and total >= .46:
    print("Congrats you secured the P based on your current estimates (if any)")
    exit()
else:
    factor = .15
    other = ""
    if paper == 0 and not skip_paper:
        factor = .3
        other += "and paper"
    for i in range(0, 501):
        score = i / 500
        if total + (factor * score) >= bins[j][1]:
            print(f"You need a {score} on the final {other} for a {bins[j][0]}")
            j -= 1
            if j == -1:
                exit()