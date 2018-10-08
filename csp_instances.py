crossword_puzzle = CSP(
    var_domains={
        # read across:
        'a1': set("bus has".split()),
        'a3': set("lane year".split()),
        'a4': set("ant car".split()),
        # read down:
        'd1': set("buys hold".split()),
        'd2': set("search syntax".split()),
    },
    constraints={
        lambda a1, d1: a1[0] == d1[0],
        lambda d1, a3: d1[2] == a3[0],
        lambda a1, d2: a1[2] == d2[0],
        lambda d2, a3: d2[2] == a3[2],
        lambda d2, a4: d2[4] == a4[0],
    })


canterbury_colouring = CSP(
    var_domains={
        'christchurch': {'red', 'green'},
        'selwyn': {'red', 'green'},
        'waimakariri': {'red', 'green'},
        },
    constraints={
        lambda christchurch, waimakariri: christchurch != waimakariri,
        lambda christchurch, selwyn: christchurch != selwyn,
        lambda selwyn, waimakariri: selwyn != waimakariri,
        })


cryptic_puzzle = CSP(
    var_domains={x: set(range(10)) for x in 'twofurabc'},
    constraints={
        lambda t, w, o, f, u, r: len({t, w, o, f, u, r}) == 6,
        lambda o, r, a: o + o == r + 10 * a,
        lambda w, u, a, b: w + w + a == u + 10 * b,
        lambda t, o, b, c: t + t + b == o + 10 * c,
        lambda f, c: c == f,
        lambda f: f != 0,
        lambda t: t != 0,
    })