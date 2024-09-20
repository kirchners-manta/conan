#!/usr/bin/python3.10


class Person:
    def __init__(self, name):
        self._name = name
        print(f"Der Name ist: {self._name}")

    @property
    def name(self):
        print(f"Name: {self._name}")
        return self._name

    @name.setter
    def name(self, new_name):
        try:
            if len(new_name) > 3:
                self._name = new_name
                print(f"Der Name wurde geändert in: {self._name}")
            else:
                raise ValueError(f"Der Name bleibt! ({new_name} ist zu kurz)")
        except ValueError as e:
            print(e)


p = Person("Sarah")
p.name = "A"
p.name = "Leo"
p.name = "Dominik"
p.name


# now i want to define a new class and inherit the class Person
class Student(Person):
    def __init__(self, name, age):
        super().__init__(name)
        self._age = age
        print(f"Das Alter ist: {self._age}")

    @property
    def age(self):
        print(f"Alter: {self._age}")
        return self._age

    @age.setter
    def age(self, new_age):
        try:
            if new_age > 0:
                self._age = new_age
                print(f"Das Alter wurde geändert in: {self._age}")
            else:
                raise ValueError(f"Das Alter bleibt! ({new_age} ist zu kurz)")
        except ValueError as e:
            print(e)


s = Student("Sarah", 20)
s.age = 0
s.age = 21
s.name = "Tina"
s.name = "Leo"
