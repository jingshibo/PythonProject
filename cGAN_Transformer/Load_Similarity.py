##
from cGAN_Transformer.Functions import Storage


##
all_subjects = {}
version = 0
result_set = 0
gen_model = 'two_factors'  # one_factor or two_factors
model_type = 'Transformer_Gan'


##
subject = 'Number0'
all_subjects[subject] = Storage.loadSimilarityResult(subject, version, result_set, model_type, gen_model)
subject = 'Number1'
all_subjects[subject] = Storage.loadSimilarityResult(subject, version, result_set, model_type, gen_model)
subject = 'Number2'
all_subjects[subject] = Storage.loadSimilarityResult(subject, version, result_set, model_type, gen_model)
subject = 'Number3'
all_subjects[subject] = Storage.loadSimilarityResult(subject, version, result_set, model_type, gen_model)
subject = 'Number4'
all_subjects[subject] = Storage.loadSimilarityResult(subject, version, result_set, model_type, gen_model)
# subject = 'Number5'
# all_subjects[subject] = Storage.loadSimilarityResult(subject, version, result_set, model_type, gen_model)
# subject = 'Number6'
# all_subjects[subject] = Storage.loadSimilarityResult(subject, version, result_set, model_type, gen_model)
subject = 'Number7'
all_subjects[subject] = Storage.loadSimilarityResult(subject, version, result_set, model_type, gen_model)
subject = 'Number8'
all_subjects[subject] = Storage.loadSimilarityResult(subject, version, result_set, model_type, gen_model)


##
import numpy as np
from scipy import stats

old_vs_new_vals = []
new_vs_fake_vals = []

for subject, data in all_subjects.items():
    comp = data["comparison"]
    old_vs_new_vals.append(comp["old_vs_new"]["overall"])
    new_vs_fake_vals.append(comp["new_vs_fake"]["overall"])

old_vs_new_vals = np.array(old_vs_new_vals)
new_vs_fake_vals = np.array(new_vs_fake_vals)


mean_old_new = np.mean(old_vs_new_vals)
std_old_new  = np.std(old_vs_new_vals, ddof=1)

mean_new_fake = np.mean(new_vs_fake_vals)
std_new_fake  = np.std(new_vs_fake_vals, ddof=1)

print("Old vs New:  mean =", mean_old_new, "std =", std_old_new)
print("New vs Fake: mean =", mean_new_fake, "std =", std_new_fake)


t_stat, p_value = stats.ttest_rel(old_vs_new_vals, new_vs_fake_vals)

print("Paired t-test:")
print("t =", t_stat)
print("p =", p_value)

diff = new_vs_fake_vals - old_vs_new_vals
cohens_d = np.mean(diff) / np.std(diff, ddof=1)

print("Cohen's d =", cohens_d)

group_stats = {
    "old_vs_new": {
        "mean": float(mean_old_new),
        "std": float(std_old_new)
    },
    "new_vs_fake": {
        "mean": float(mean_new_fake),
        "std": float(std_new_fake)
    },
    "paired_t_test": {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d)
    }
}