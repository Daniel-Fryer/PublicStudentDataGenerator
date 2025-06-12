# Note: This script is fully portable and requires only 'StudentDataInitialDataSet.csv' 
# to be present in the same folder as the script. Output will be created in a folder 
# called 'SavedProblemInstances' in the same location.

import os
import pandas as pd
import random
import string
import numpy as np

# Load ethnicity source dataset from the current directory
sourceFile = "StudentDataInitialDataSet.csv"
if not os.path.exists(sourceFile):
    raise FileNotFoundError(
        f"Required data file '{sourceFile}' not found in current directory: {os.getcwd()}"
    )
df = pd.read_csv(sourceFile)
ethnicities = df['Ethnicity'].dropna().unique().tolist()

def generateNamePool(numNames, length=8):
    return [''.join(random.choices(string.ascii_letters, k=length)).capitalize() for _ in range(numNames)]

def generateStudentData(numStudents=160, genderType="Mixed"):
    duplicateFirstPct = random.uniform(0, 0.15)
    duplicateSurnamePct = random.uniform(0, 0.05)

    numFirstnames = int(numStudents * (1 - duplicateFirstPct))
    numSurnames = int(numStudents * (1 - duplicateSurnamePct))

    uniqueFirstnames = generateNamePool(numFirstnames)
    uniqueSurnames = generateNamePool(numSurnames)

    firstnames = uniqueFirstnames + random.choices(uniqueFirstnames, k=numStudents - numFirstnames)
    surnames = uniqueSurnames + random.choices(uniqueSurnames, k=numStudents - numSurnames)

    random.shuffle(firstnames)
    random.shuffle(surnames)

    fullNames = set()
    names = []
    i = 0
    while len(names) < numStudents:
        name = f"{firstnames[i]}, {surnames[i]}"
        if name not in fullNames:
            names.append(name)
            fullNames.add(name)
        else:
            surnames[i] = ''.join(random.choices(string.ascii_letters, k=8)).capitalize()
        i += 1

    # Gender assignment
    if genderType == "F":
        genders = ['F'] * numStudents
    elif genderType == "M":
        genders = ['M'] * numStudents
    else:
        genderSplit = random.randint(35, 65)
        numMales = round(numStudents * genderSplit / 100)
        numFemales = numStudents - numMales
        if random.choice([True, False]):
            genders = ['M'] * numMales + ['F'] * numFemales
        else:
            genders = ['M'] * numFemales + ['F'] * numMales
        random.shuffle(genders)

    # FSM assignment
    fsmBase = 0.28
    fsmVar = 0.15
    fsmPct = random.uniform(fsmBase - fsmVar, fsmBase + fsmVar)  # [0.13, 0.43]
    numFsm = int(numStudents * fsmPct)
    fsmIndices = random.sample(range(numStudents), numFsm)
    fsmList = ['N'] * numStudents
    for idx in fsmIndices:
        fsmList[idx] = 'Y'

    # PP assignment: 80% of FSM are PP + extra 20–30% of FSM count, randomly assigned among non-FSM 
    numPpFromFsm = int(numFsm * 0.8)
    fsmYesIndices = [idx for idx in fsmIndices]
    ppFromFsm = random.sample(fsmYesIndices, numPpFromFsm)

    numExtraPp = int(numFsm * random.uniform(0.2, 0.3))
    nonFsmIndices = [i for i in range(numStudents) if fsmList[i] == 'N']
    extraPpIndices = random.sample(nonFsmIndices, numExtraPp)

    ppList = ['N'] * numStudents
    for idx in ppFromFsm:
        ppList[idx] = 'Y'
    for idx in extraPpIndices:
        ppList[idx] = 'Y'

    midyis = np.clip(np.random.normal(loc=100, scale=15, size=numStudents), 50, 140).round().astype(int)

    # SEN assignment with MIDYIS-aligned weighting and fixed overall probability
    sen = []
    senRates = []  # Temporary list to determine weighted choices later

    for score in midyis:
        if score < 65:
            senRates.append((0.3, 0.4, 0.2))  # E, K, blank
        elif score < 90:
            senRates.append((0.2, 0.5, 0.3))
        else:
            senRates.append((0.02, 0.12, 0.86))

    # Total number of SEN students to assign (between 5-25%)
    totalSEN = random.randint(int(numStudents * 0.05), int(numStudents * 0.25))
    senIndices = random.sample(range(numStudents), totalSEN)

    for i in range(numStudents):
        if i in senIndices:
            roll = random.random()
            pE, pK, _ = senRates[i]
            if roll < pE:
                sen.append('E')
            elif roll < pE + pK:
                sen.append('K')
            else:
                sen.append(np.nan)
        else:
            sen.append(np.nan)

    # Map each unique ethnicity to its group
    ethnicityToGroup = {}
    for eth in ethnicities:
        ethLower = eth.lower()
        if "white" in ethLower and "british" in ethLower:
            ethnicityToGroup[eth] = "whiteBritish"
        elif any(x in ethLower for x in ["white other", "irish", "italian"]):
            ethnicityToGroup[eth] = "otherWhite"
        elif any(x in ethLower for x in ["black", "african", "caribbean"]):
            ethnicityToGroup[eth] = "black"
        elif any(x in ethLower for x in ["asian", "indian", "chinese"]):
            ethnicityToGroup[eth] = "asian"
        elif any(x in ethLower for x in ["mixed", "white and asian"]):
            ethnicityToGroup[eth] = "mixed"
        elif "refused" in ethLower or "unknown" in ethLower:
            ethnicityToGroup[eth] = "refused"
        else:
            ethnicityToGroup[eth] = "other"

    # Invert: group to list of ethnicities
    groupToEthnicities = {}
    for eth, grp in ethnicityToGroup.items():
        groupToEthnicities.setdefault(grp, []).append(eth)

    # Defined ranges for each group
    ranges = {
        "whiteBritish": (0.60, 1.00),
        "otherWhite":   (0.00, 0.15),
        "black":        (0.00, 0.10),
        "asian":        (0.00, 0.15),
        "mixed":        (0.00, 0.15),
        "other":        (0.00, 0.10),
        "refused":      (0.00, 0.10)
    }

    # Draw random fractions within the above ranges, ensuring sum <= 1
    whiteBritishFrac = random.uniform(*ranges["whiteBritish"])
    remainder = 1.0 - whiteBritishFrac

    # Sequentially assign other groups
    fracOtherWhite = random.uniform(*[ranges["otherWhite"][0], min(ranges["otherWhite"][1], remainder)])
    remainder -= fracOtherWhite

    fracBlack = random.uniform(*[ranges["black"][0], min(ranges["black"][1], remainder)])
    remainder -= fracBlack

    fracAsian = random.uniform(*[ranges["asian"][0], min(ranges["asian"][1], remainder)])
    remainder -= fracAsian

    fracMixed = random.uniform(*[ranges["mixed"][0], min(ranges["mixed"][1], remainder)])
    remainder -= fracMixed

    fracOther = random.uniform(*[ranges["other"][0], min(ranges["other"][1], remainder)])
    remainder -= fracOther

    # Convert fractions to student counts
    numWhiteBritish = int(numStudents * whiteBritishFrac)
    numOtherWhite   = int(numStudents * fracOtherWhite)
    numBlack        = int(numStudents * fracBlack)
    numAsian        = int(numStudents * fracAsian)
    numMixed        = int(numStudents * fracMixed)
    numOther        = int(numStudents * fracOther)
    numRefused      = numStudents - (numWhiteBritish + numOtherWhite + numBlack +
                                     numAsian + numMixed + numOther)

    groupSizes = {
        "whiteBritish": numWhiteBritish,
        "otherWhite":   numOtherWhite,
        "black":        numBlack,
        "asian":        numAsian,
        "mixed":        numMixed,
        "other":        numOther,
        "refused":      numRefused
    }

    # Guarantee every original ethnicity is represented at least once
    minEthnicityRepresentations = []
    for eth in ethnicities:
        minEthnicityRepresentations.append(eth)

    # Distribute the remaining students
    ethnicityList = minEthnicityRepresentations.copy()
    for group, size in groupSizes.items():
        groupSize = size - sum(1 for eth in minEthnicityRepresentations if ethnicityToGroup[eth] == group)
        if groupSize > 0 and groupToEthnicities.get(group):
            ethnicityList += random.choices(groupToEthnicities[group], k=groupSize)

    random.shuffle(ethnicityList)
    ethnicitiesList = ethnicityList[:numStudents]

    syntheticDf = pd.DataFrame({
        'Name': names,
        'Gender': genders,
        'FSM': fsm,
        'SEN Status': sen,
        'Ethnicity': ethnicitiesList,
        'Pupil Premium Indicator': pp,
        'MIDYIS Mean': midyis
    })

    return syntheticDf

def generateStudentPairs(numStudents, goodPairPctRange=(0, 0.25), badPairPctRange=(0, 0.10)):
    # Each pair is two unique student indices
    allIndices = list(range(numStudents))
    # Good pairs
    maxGoodPairs = int(numStudents * random.uniform(*goodPairPctRange))
    goodPairs = set()
    while len(goodPairs) < maxGoodPairs:
        i, j = random.sample(allIndices, 2)
        pair = tuple(sorted([i, j]))
        if pair not in goodPairs:
            goodPairs.add(pair)
    # Bad pairs
    maxBadPairs = int(numStudents * random.uniform(*badPairPctRange))
    badPairs = set()
    while len(badPairs) < maxBadPairs:
        i, j = random.sample(allIndices, 2)
        pair = tuple(sorted([i, j]))
        if pair not in badPairs:
            badPairs.add(pair)
    return list(goodPairs), list(badPairs)

def writePairDataCsv(outputFile, goodPairs, badPairs):
    import csv
    with open(outputFile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PairType', 'Student1', 'Student2'])
        for i, j in goodPairs:
            writer.writerow(['Good', i, j])
        for i, j in badPairs:
            writer.writerow(['Bad', i, j])

def getNextInstanceFilename(baseDir, sizeTag, genderTag, prefix="StudentDataset", ext=".csv"):
    i = 1
    while True:
        filename = f"{prefix}{sizeTag}{genderTag}_Instance{i}{ext}"
        candidate = os.path.join(baseDir, filename)
        if not os.path.exists(candidate):
            return candidate
        i += 1

if __name__ == "__main__":
    numToCreate = 10
    genderType = "Mixed"  # "F", "M", or "Mixed"
    numStudents = 160     # Student count per dataset

    if genderType == "F":
        genderTag = "_AllFemale"
    elif genderType == "M":
        genderTag = "_AllMale"
    else:
        genderTag = "_MixedGender"

    sizeTag = f"_n={numStudents}"

    outputDir = os.path.join(os.getcwd(), "SavedProblemInstances")
    os.makedirs(outputDir, exist_ok=True)

    for instanceIndex in range(numToCreate):
        syntheticData = generateStudentData(numStudents=numStudents, genderType=genderType)
        outputFile = getNextInstanceFilename(outputDir, sizeTag, genderTag)
        syntheticData.to_csv(outputFile, index=False)
        print(f"Synthetic data saved to: {outputFile}")

        # Generate pairs and save to pair file
        goodPairs, badPairs = generateStudentPairs(numStudents)
        pairFile = outputFile.replace('.csv', '_PairData.csv')
        writePairDataCsv(pairFile, goodPairs, badPairs)
        print(f"Pair data saved to: {pairFile}")
