# Rules Summary - Coverage, Utility, and Prevalence


### GERMAN Dataset

| Rule | Condition | Treatment | Coverage (%) | Utility | Prevalence (%) |
|:----:|-----------|-----------|:------------:|:-------:|:--------------:|
| 1 | foreign_worker = yes | status → no checking account | 96.3 | 0.1609 | 8.98 |
| 2 | foreign_worker = yes | present_residence → 4 | 96.3 | 0.0606 | 19.57 |
| 3 | people_liable = 1 | status → no checking account | 84.5 | 0.1457 | 22.7 |
| 4 | people_liable = 1 | present_residence → 4 | 84.5 | 0.0793 | 14.34 |
| 5 | foreign_worker = yes | job → skilled employee/official | 96.3 | 0.0439 | 20.0 |
| 6 | amount = 0 | present_residence → 4 | 75.4 | 0.0583 | 27.25 |
| 7 | amount = 0 | status → no checking account | 75.4 | 0.2666 | 8.63 |
| 8 | foreign_worker = yes | savings → unknown/no savings account | 96.3 | 0.1364 | 8.24 |
| 9 | foreign_worker = yes | number_credits → 1 | 96.3 | 0.0142 | 8.47 |
| 10 | people_liable = 1 | savings → unknown/no savings account | 84.5 | 0.1299 | 17.95 |


### SO Dataset

| Rule | Condition | Treatment | Coverage (%) | Utility | Prevalence (%) |
|:----:|-----------|-----------|:------------:|:-------:|:--------------:|
| 1 | Gender = Male | Exercise → Daily or almost every day | 92.4 | 11626.767 | 8.98 |
| 2 | SexualOrientation = Straight or heterosexual | FormalEducation → Other doctoral degree (Ph.D, Ed.D., etc.) | 92.89 | 17285.5265 | 19.57 |
| 3 | Student = No | DevType → Engineering manager | 83.46 | 43992.6199 | 22.7 |
| 4 | RaceEthnicity = White or of European descent | UndergradMajor → A natural science (ex. biology, chemistry, physics) | 77.63 | 7078.5843 | 14.34 |
| 5 | Student = No | FormalEducation → Other doctoral degree (Ph.D, Ed.D., etc.) | 83.46 | 26000.059 | 20.0 |
| 6 | RaceEthnicity = White or of European descent | DevType → Engineering manager | 77.63 | 57551.4186 | 27.25 |
| 7 | Student = No | Exercise → 3 - 4 times per week | 83.46 | 23905.4004 | 8.63 |
| 8 | SexualOrientation = Straight or heterosexual | Exercise → 3 - 4 times per week | 92.89 | 23651.2064 | 8.24 |
| 9 | Dependents = No | HoursComputer → 5 - 8 hours | 69.76 | 16564.594 | 8.47 |
| 10 | Gender = Male | UndergradMajor → A humanities discipline (ex. literature, history, philosophy) | 92.4 | 16320.1586 | 17.95 |


### ACS Dataset

| Rule | Condition | Treatment | Coverage (%) | Utility | Prevalence (%) |
|:----:|-----------|-----------|:------------:|:-------:|:--------------:|
| 1 | With a disability = 2 | Insurance through a current or former employer or union → 1 | 84.04 | 15278.8312 | 8.98 |
| 2 | With a disability = 2 | Public health coverage → 2 | 84.04 | 7259.4758 | 19.57 |
| 3 | With a disability = 2 | Private health insurance coverage → 2 | 84.04 | 1302.4318 | 22.7 |
| 4 | With a disability = 2 | Educational attainment → 20 | 84.04 | 2602.1199 | 14.34 |
| 5 | With a disability = 2 | Educational attainment → 21 | 84.04 | 10610.2679 | 20.0 |
| 6 | With a disability = 2 | Educational attainment → 22 | 84.04 | 14823.5245 | 27.25 |
| 7 | Sex = 2 | Educational attainment → 20 | 51.54 | 3195.9216 | 8.63 |
| 8 | Sex = 1 | Educational attainment → 21 | 48.46 | 12832.1631 | 8.24 |
| 9 | Sex = 1 | Educational attainment → 20 | 48.46 | 2950.5532 | 8.47 |
| 10 | Sex = 1 | Insurance through a current or former employer or union → 1 | 48.46 | 16872.0595 | 17.95 |

