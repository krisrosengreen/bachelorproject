import os
from main import FILENAME, FILEOUTPUT, PP_FILENAME, PP_FILEOUTPUT

def get_string_within(s, char1, char2):
    string_builder = ""

    start = -1
    end = -1

    for i in range(len(s)):
        if s[i] == char1:
            start = i

        if s[i] == char2:
            end = i
            break

    return s[start+1:end]

def check_eigenvalues(filename) -> bool:  # There has to be 8 eigenvalues
    with open(filename, 'r') as f:
        file_content = f.read()

        lines = file_content.split("\n")
        energies = list(filter(lambda s: "e(" in s, lines))

        current = 0
        bad_value = False

        for index, energy in enumerate(energies):
            energy_eigenvalue_count = get_string_within(energy, '(', ')')
            count1_char = energy_eigenvalue_count.strip()[0]
            count2_char = energy_eigenvalue_count.strip()[-1]

            count1 = int(count1_char)
            count2 = int(count2_char)

            if count1 == current+1 or count2 == current+1:
                current+= count2-count1
                current+=1
            else:
                if not bad_value:
                    print("Bad eigenvalue!")
                    print("...")
                    print("\n".join(energies[index-5:index]))
                    print(energies[index], count1, count2, "expected", current+1 ,"\r **")
                    print("\n".join(energies[index+1:index+5]))
                    print("...")
                    bad_value = True
                else:
                    bad_value = False
                    current = max(count1, count2)
            if current == 8 or count2 == 8:
                current = 0
    return not bad_value


def check_success(espresso_output) -> bool:
    return "JOB DONE" in espresso_output


def calculate_energies() -> bool:  # Returns True if successful
    outp1 = os.popen(f"pw.x -i {FILENAME} > {FILEOUTPUT}; cat {FILEOUTPUT}")
    outp2 = os.popen(f"bands.x < {PP_FILENAME} > {PP_FILEOUTPUT}; cat {PP_FILEOUTPUT}")

    check1 = check_success(outp1.read())
    check2 = check_success(outp2.read())
    check3 = check_eigenvalues("si_bands_pp.out")

    return check1 and check2 and check3  # Check if all were successful


if __name__ == "__main__":
    check_eigenvalues("si_bands_pp_py.out")
