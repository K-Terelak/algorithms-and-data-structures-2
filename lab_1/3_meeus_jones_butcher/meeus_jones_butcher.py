from datetime import datetime


def meeus_jones_butcher_easter_date(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    p = h + l - 7 * m + 114
    month = p // 31
    day = (p % 31) + 1

    return datetime(year, month, day)


if __name__ == '__main__':
    years = [2190, 1818]

    for year in years:
        easter = meeus_jones_butcher_easter_date(year)
        print(f"{easter.strftime('%d %B %Y')}")
