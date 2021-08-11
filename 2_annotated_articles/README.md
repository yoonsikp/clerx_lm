## Annotation Format
The filename contains the PMID, as well as a list of the matched regex. The first line is the article title, every line after is from the abstract. The annotations use tab separated values (TSV).

Each annotated sentence will have the following columns:
`sentence_text<TAB>entities<TAB>true_relations`

## Entity Annotation Format
Each entity label has an incrementing ID starting from 1

Entity labels: `explanatory_var, outcome_var, hr_ratio, or_ratio, rr_ratio, baseline`

Entity label IDs: `1,2,3,4,5,6`

Each entity annotation consists of the following
```
label_id,begin,end
```

Example:
```
1,23,34
```

Multiple entity annotations are seperated using a period.
```
1,23,34.3,56,60
```

## Relation Annotation Format
List of true relations between entities. Entities are identified from the previous column, in order, starting at one.

Example:
```
1,2
```

Multiple true relations are seperated using a period. Assuming there were 3 entities in the previous column, for example:
```
1,3.2,3
```

## Annotation Guidelines
We annotated all explanatory variables, even those without an associated ratio in the sentence

We did not annotate the word "patients", "outcome", "burden", or "diagnosis" unless necessary

Prefixed adjectives that unnecessarily modify the outcome variable were not included, e.g.
NO: (favorable functional outcome at 3 months) YES: (functional outcome at 3 months)

We omited articles such as "the" or "a".

Sometimes outcome and explanatory variables could not be differentiated, i.e. when it's a population characteristic. This is because P(O|AC)/P(O|BC) = P(O∩C|A)/P(O∩C|B).
