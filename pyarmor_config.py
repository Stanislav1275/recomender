from pyarmor import pack

# Укажите путь к вашему проекту
project_path = '.'

# Укажите папку, которую нужно исключить
exclude = ['env', '.env', 'env.example', 'gitignore']

# Упаковка проекта
pack(project_path, exclude=exclude)