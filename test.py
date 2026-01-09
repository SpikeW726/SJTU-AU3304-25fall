def gen_test(x):
    for i in x:
        yield i

result = gen_test("wodhqih")
print(result)
print(next(result))

