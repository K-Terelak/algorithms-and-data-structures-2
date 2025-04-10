from datetime import datetime


def gauss_easter_date(year):
    a = year % 19
    b = year % 4
    c = year % 7
    k = year // 100
    p = (13 + 8 * k) // 25
    q = k // 4

    A = (15 - p + k - q) % 30
    B = (4 + k - q) % 7

    d = (19 * a + A) % 30
    e = (2 * b + 4 * c + 6 * d + B) % 7

    if d == 29 and e == 6:
        return datetime(year, 4, 19)
    elif d == 28 and e == 6 and a > 10:
        return datetime(year, 4, 18)

    if (d + e) <= 9:
        return datetime(year, 3, 22 + d + e)
    else:
        return datetime(year, 4, d + e - 9)


if __name__ == '__main__':
    years = [2170, 2049, 2331, 1609]

    for year in years:
        easter_date = gauss_easter_date(year)
        print(f"{easter_date.strftime('%d %B %Y')}")
