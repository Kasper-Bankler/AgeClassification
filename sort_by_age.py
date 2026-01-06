import os
import shutil

# 🔧 JUSTÉR DISSE
SOURCE_DIR = r"C:\Users\victo\Desktop\Programmering eksamen\AgeClassification\data\UTKFace"
TARGET_DIR = r"C:\Users\victo\Desktop\Programmering eksamen\AgeClassification\data\age_groups"

AGE_STEP = 5   # 0-4, 5-9, 10-14 osv.
MAX_AGE = 100 # alt over ryger i sidste gruppe

os.makedirs(TARGET_DIR, exist_ok=True)

def age_to_group(age):
    if age >= MAX_AGE:
        return f"{MAX_AGE}+"
    low = (age // AGE_STEP) * AGE_STEP
    high = low + AGE_STEP - 1
    return f"{low}-{high}"

count = 0
for filename in os.listdir(SOURCE_DIR):
    if not filename.lower().endswith(".jpg"):
        continue

    try:
        age = int(filename.split("_")[0])
    except ValueError:
        print("Springer over:", filename)
        continue

    group = age_to_group(age)
    group_dir = os.path.join(TARGET_DIR, group)
    os.makedirs(group_dir, exist_ok=True)

    src = os.path.join(SOURCE_DIR, filename)
    dst = os.path.join(group_dir, filename)
    shutil.copy(src, dst)  # brug move() hvis du vil flytte

    count += 1

print(f"✔ Sorteret {count} billeder i alders-grupper")
