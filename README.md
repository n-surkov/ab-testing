# ab-testing

Проект сравнения некоторых методов проверки гипотез АБ-тестов.

* [abtools](./abtools) -- пакет с реализованными методами проверки
* [methods_comparison](./methods_comparison.ipynb) -- демонстрация использования методов и их сравнение

## Методы проверки гипотез можно установить пакетом
### Remote

```bash
$ python3 -m pip install git+ssh://git@github.com:n-surkov/ab-testing.git
```

### Local

```bash
$ git clone git@github.com:n-surkov/ab-testing.git
$ python3 -m pip install -e ./ab-testing
```