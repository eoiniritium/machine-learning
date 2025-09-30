"""Process data downloaded from Kaggle (https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval)"""

def processLine(commaSeperated, isTest=False):

    # Extract data
    isMale = '1' if commaSeperated[1] == 'Male' else '0'
    isMarried = '1' if commaSeperated[2] == 'Yes' else '0'
    dependents = commaSeperated[3] if commaSeperated[3] else '0'
    isGraduate = '1' if commaSeperated[4] == 'Graduate' else '0'
    match commaSeperated[5]:
        case 'Yes':
            employment = '1'
        case 'Yes':
            employment = '0'
        case _:
            employment = '-1'
    income1 = commaSeperated[6]
    income2 = commaSeperated[7]
    loanAmount = commaSeperated[8] if commaSeperated[8] else '0'
    loanTerm = commaSeperated[9] if commaSeperated[9] else '0'
    creditHistory = commaSeperated[10] if commaSeperated[10] else '0'
    match commaSeperated[11]:
        case 'Urban':
            propertyArea = '2'
        case 'Semiurban':
            propertyArea = '1'
        case 'Rural':
            propertyArea = '0'
    if isTest:
        isApproved = '1' # Dummy - not used for prediction
    else:
        isApproved = '1' if commaSeperated[12] == 'Y' else '0'

    # Normalise data
    loanAmount = str(float(loanAmount)/10000.0)
    income1 = str(float(income1)/10000.0)
    income2 = str(float(income2)/10000.0)
    loanTerm = str(float(loanTerm)/180)


    outputLine = ','.join([
        isMale, isMarried, dependents, isGraduate, employment, income1, income2, loanAmount, loanTerm, creditHistory, propertyArea
    ]) + ' ' + isApproved

    return outputLine



linesOut = []

with open('loan_sanction_train.csv', 'r') as linesIn:
    for i, lineIn in enumerate(linesIn):
        if i == 0:
            continue

        commaSeperated = [elem.strip() for elem in lineIn.split(',')]

        processedLine = processLine(commaSeperated, isTest=False)

        if processedLine:
            linesOut.append(processLine(commaSeperated, isTest=False))


with open('homeloan-train.txt', 'w') as file:
    file.write('\n'.join(linesOut))