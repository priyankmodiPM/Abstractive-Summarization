test_input = open("data/gigaword/Giga/input.txt")
test_target = open("data/gigaword/Giga/task1_ref0.txt")

test_input_cleaned = open("data/gigaword/Giga/input_cleaned.txt", "w")
test_target_cleaned = open("data/gigaword/Giga/task1_ref0_cleaned.txt", "w")

for source, target in zip(test_input.readlines(), test_target.readlines()):
    if source.strip() == "UNK":
        continue
    test_input_cleaned.write(source)
    test_target_cleaned.write(target)

test_input.close()
test_target.close()
test_input_cleaned.close()
test_target_cleaned.close()