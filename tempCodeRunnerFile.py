while True:
    try:
        password = input()

        if len(password) <= 8:
            print("NG")
        else:
            res = [0, 0, 0, 0]
            i = 0
            flag = True
            while i < len(password):
                char = password[i]
                if i < len(password) - 4:
                    if password[i+3:].find(password[i:i+3]) != -1:
                        print("NG")
                        flag = False
                        break
                if "a" <= char <= "z":
                    if res[0] == 0:
                        res[0] += 1
                elif "A" <= char <= "Z":
                    if res[1] == 0:
                        res[1] += 1
                elif "0" <= char <= "9":
                    if res[2] == 0:
                        res[2] += 1
                elif res[3] == 0:
                    res[3] += 1
                i += 1
            
            if flag:
                if sum(res) >= 3:
                    print("OK")
                else:
                    print("NG")
    except:
        break