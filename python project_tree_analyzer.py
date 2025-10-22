#!/usr/bin/env python3
"""
Анализатор структуры проекта медицинского ассистента
Сканирует директории и файлы, выводит дерево проекта с анализом
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

class ProjectTreeAnalyzer:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.ignore_dirs = {
            '__pycache__', '.git', '.vscode', '.idea', 'node_modules', 
            'venv', 'env', '.env', 'dist', 'build', '.pytest_cache',
            '.mypy_cache', '.tox', 'htmlcov', '.coverage'
        }
        self.ignore_files = {
            '.pyc', '.pyo', '.pyd', '.so', '.egg-info', '.git',
            '.DS_Store', 'Thumbs.db', '.gitignore', '.gitkeep'
        }
        
        # Категории файлов для анализа
        self.file_categories = {
            'python': ['.py'],
            'config': ['.yaml', '.yml', '.json', '.toml', '.ini', '.cfg', '.env'],
            'docs': ['.md', '.rst', '.txt', '.pdf'],
            'data': ['.csv', '.json', '.xml', '.sql', '.db', '.sqlite'],
            'media': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.dcm'],
            'web': ['.html', '.css', '.js', '.tsx', '.jsx'],
            'requirements': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile']
        }

    def get_file_category(self, file_path: Path) -> str:
        """Определяет категорию файла"""
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()
        
        for category, extensions in self.file_categories.items():
            if suffix in extensions or name in extensions:
                return category
        return 'other'

    def analyze_python_file(self, file_path: Path) -> Dict:
        """Анализирует Python файл"""
        info = {
            'lines': 0,
            'imports': [],
            'classes': [],
            'functions': [],
            'streamlit_pages': [],
            'ai_models': [],
            'medical_terms': []
        }
        
        medical_keywords = [
            'ecg', 'ekg', 'xray', 'mri', 'ct', 'диагноз', 'пациент', 'patient',
            'diagnosis', 'medical', 'clinical', 'анализ', 'protocol', 'протокол'
        ]
        
        ai_keywords = [
            'claude', 'openai', 'anthropic', 'openrouter', 'ai', 'model',
            'vision', 'chat', 'assistant', 'ии', 'нейро'
        ]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                info['lines'] = len(lines)
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Импорты
                    if line.startswith(('import ', 'from ')):
                        info['imports'].append(line)
                    
                    # Классы
                    elif line.startswith('class '):
                        class_name = line.split('(')[0].replace('class ', '').strip(':')
                        info['classes'].append(class_name)
                    
                    # Функции
                    elif line.startswith('def '):
                        func_name = line.split('(')[0].replace('def ', '')
                        info['functions'].append(func_name)
                    
                    # Streamlit страницы
                    elif 'st.' in line and any(x in line for x in ['header', 'title', 'subheader']):
                        info['streamlit_pages'].append(f"Line {line_num}: {line[:50]}...")
                    
                    # ИИ модели
                    line_lower = line.lower()
                    for keyword in ai_keywords:
                        if keyword in line_lower:
                            info['ai_models'].append(f"Line {line_num}: {line[:50]}...")
                            break
                    
                    # Медицинские термины
                    for keyword in medical_keywords:
                        if keyword in line_lower:
                            info['medical_terms'].append(f"Line {line_num}: {line[:50]}...")
                            break
                            
        except Exception as e:
            info['error'] = str(e)
            
        return info

    def scan_directory(self, path: Path, max_depth: int = 5, current_depth: int = 0) -> Dict:
        """Сканирует директорию рекурсивно"""
        if current_depth > max_depth:
            return {}
            
        result = {
            'type': 'directory',
            'name': path.name,
            'path': str(path),
            'children': {},
            'stats': {
                'total_files': 0,
                'total_dirs': 0,
                'python_files': 0,
                'total_lines': 0,
                'categories': {}
            }
        }
        
        try:
            for item in sorted(path.iterdir()):
                # Пропускаем игнорируемые
                if item.name in self.ignore_dirs:
                    continue
                if any(item.name.endswith(ext) for ext in self.ignore_files):
                    continue
                    
                if item.is_file():
                    category = self.get_file_category(item)
                    file_info = {
                        'type': 'file',
                        'name': item.name,
                        'path': str(item),
                        'size': item.stat().st_size,
                        'category': category
                    }
                    
                    # Анализ Python файлов
                    if category == 'python':
                        py_analysis = self.analyze_python_file(item)
                        file_info['analysis'] = py_analysis
                        result['stats']['python_files'] += 1
                        result['stats']['total_lines'] += py_analysis.get('lines', 0)
                    
                    result['children'][item.name] = file_info
                    result['stats']['total_files'] += 1
                    
                    # Статистика по категориям
                    if category not in result['stats']['categories']:
                        result['stats']['categories'][category] = 0
                    result['stats']['categories'][category] += 1
                    
                elif item.is_dir():
                    subdir = self.scan_directory(item, max_depth, current_depth + 1)
                    if subdir:  # Если директория не пустая
                        result['children'][item.name] = subdir
                        result['stats']['total_dirs'] += 1
                        
                        # Суммируем статистику
                        sub_stats = subdir['stats']
                        result['stats']['total_files'] += sub_stats['total_files']
                        result['stats']['total_dirs'] += sub_stats['total_dirs']
                        result['stats']['python_files'] += sub_stats['python_files']
                        result['stats']['total_lines'] += sub_stats['total_lines']
                        
                        for cat, count in sub_stats['categories'].items():
                            if cat not in result['stats']['categories']:
                                result['stats']['categories'][cat] = 0
                            result['stats']['categories'][cat] += count
                            
        except PermissionError:
            result['error'] = 'Permission denied'
            
        return result

    def print_tree(self, tree_data: Dict, prefix: str = "", is_last: bool = True, show_analysis: bool = False):
        """Выводит дерево в консоль"""
        if not tree_data:
            return
            
        # Символы для дерева
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{tree_data['name']}")
        
        # Показываем статистику для директорий
        if tree_data['type'] == 'directory' and tree_data['stats']['total_files'] > 0:
            stats = tree_data['stats']
            next_prefix = prefix + ("    " if is_last else "│   ")
            print(f"{next_prefix}📊 Files: {stats['total_files']}, "
                  f"Python: {stats['python_files']}, "
                  f"Lines: {stats['total_lines']:,}")
            
            if stats['categories']:
                categories_str = ", ".join([f"{cat}: {count}" for cat, count in stats['categories'].items()])
                print(f"{next_prefix}📁 {categories_str}")
        
        # Показываем анализ Python файлов
        if (show_analysis and tree_data['type'] == 'file' and 
            tree_data['category'] == 'python' and 'analysis' in tree_data):
            
            analysis = tree_data['analysis']
            next_prefix = prefix + ("    " if is_last else "│   ")
            
            if analysis.get('lines', 0) > 0:
                print(f"{next_prefix}🐍 Lines: {analysis['lines']}")
            
            if analysis.get('classes'):
                print(f"{next_prefix}🏗️  Classes: {', '.join(analysis['classes'][:3])}...")
            
            if analysis.get('functions'):
                print(f"{next_prefix}⚙️  Functions: {len(analysis['functions'])} found")
            
            if analysis.get('ai_models'):
                print(f"{next_prefix}🤖 AI: {len(analysis['ai_models'])} references")
            
            if analysis.get('medical_terms'):
                print(f"{next_prefix}🏥 Medical: {len(analysis['medical_terms'])} terms")
        
        # Рекурсивно обрабатываем детей
        if 'children' in tree_data:
            children = list(tree_data['children'].items())
            for i, (name, child) in enumerate(children):
                is_last_child = i == len(children) - 1
                child_prefix = prefix + ("    " if is_last else "│   ")
                self.print_tree(child, child_prefix, is_last_child, show_analysis)

    def generate_summary(self, tree_data: Dict) -> Dict:
        """Генерирует сводку по проекту"""
        summary = {
            'project_name': tree_data['name'],
            'total_files': tree_data['stats']['total_files'],
            'total_directories': tree_data['stats']['total_dirs'],
            'python_files': tree_data['stats']['python_files'],
            'total_lines_of_code': tree_data['stats']['total_lines'],
            'file_categories': tree_data['stats']['categories'],
            'key_files': [],
            'architecture_analysis': {
                'main_modules': [],
                'ai_components': [],
                'medical_modules': [],
                'ui_components': [],
                'data_storage': []
            }
        }
        
        # Анализируем ключевые файлы
        self._analyze_key_files(tree_data, summary)
        
        return summary

    def _analyze_key_files(self, tree_data: Dict, summary: Dict, path: str = ""):
        """Анализирует ключевые файлы проекта"""
        if tree_data['type'] == 'file':
            file_path = path + "/" + tree_data['name'] if path else tree_data['name']
            
            # Главные файлы
            if tree_data['name'] in ['main.py', 'app.py', 'streamlit_app.py']:
                summary['key_files'].append(f"🚀 Main: {file_path}")
            
            # Python файлы с анализом
            if (tree_data['category'] == 'python' and 'analysis' in tree_data):
                analysis = tree_data['analysis']
                
                # ИИ компоненты
                if analysis.get('ai_models'):
                    summary['architecture_analysis']['ai_components'].append(file_path)
                
                # Медицинские модули
                if analysis.get('medical_terms'):
                    summary['architecture_analysis']['medical_modules'].append(file_path)
                
                # UI компоненты (Streamlit)
                if analysis.get('streamlit_pages'):
                    summary['architecture_analysis']['ui_components'].append(file_path)
                
                # Основные модули (много классов/функций)
                if len(analysis.get('classes', [])) > 2 or len(analysis.get('functions', [])) > 5:
                    summary['architecture_analysis']['main_modules'].append(file_path)
            
            # Файлы данных
            elif tree_data['category'] in ['data', 'config']:
                summary['architecture_analysis']['data_storage'].append(file_path)
                
        elif tree_data['type'] == 'directory' and 'children' in tree_data:
            for name, child in tree_data['children'].items():
                new_path = path + "/" + tree_data['name'] if path else tree_data['name']
                self._analyze_key_files(child, summary, new_path)

    def save_analysis(self, tree_data: Dict, summary: Dict, output_file: str = "project_analysis.json"):
        """Сохраняет анализ в JSON"""
        analysis_data = {
            'tree_structure': tree_data,
            'summary': summary,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        return output_file

def main():
    """Главная функция"""
    print("🔍 Анализатор структуры медицинского проекта")
    print("=" * 50)
    
    # Получаем путь проекта
    project_path = input("Введите путь к проекту (Enter для текущей директории): ").strip()
    if not project_path:
        project_path = "."
    
    # Создаем анализатор
    analyzer = ProjectTreeAnalyzer(project_path)
    
    # Сканируем проект
    print(f"\n🔎 Сканирование проекта: {os.path.abspath(project_path)}")
    tree_data = analyzer.scan_directory(Path(project_path))
    
    if not tree_data:
        print("❌ Проект не найден или пуст")
        return
    
    # Выводим дерево
    print(f"\n🌳 Структура проекта:")
    print("=" * 50)
    analyzer.print_tree(tree_data, show_analysis=True)
    
    # Генерируем сводку
    summary = analyzer.generate_summary(tree_data)
    
    print(f"\n📊 СВОДКА ПО ПРОЕКТУ")
    print("=" * 50)
    print(f"📁 Всего файлов: {summary['total_files']}")
    print(f"📂 Всего директорий: {summary['total_directories']}")
    print(f"🐍 Python файлов: {summary['python_files']}")
    print(f"📝 Строк кода: {summary['total_lines_of_code']:,}")
    
    print(f"\n📋 Категории файлов:")
    for category, count in summary['file_categories'].items():
        print(f"  • {category}: {count}")
    
    print(f"\n🏗️ АРХИТЕКТУРНЫЙ АНАЛИЗ:")
    arch = summary['architecture_analysis']
    
    if arch['main_modules']:
        print(f"\n🚀 Основные модули:")
        for module in arch['main_modules'][:5]:
            print(f"  • {module}")
    
    if arch['ai_components']:
        print(f"\n🤖 ИИ компоненты:")
        for component in arch['ai_components'][:5]:
            print(f"  • {component}")
    
    if arch['medical_modules']:
        print(f"\n🏥 Медицинские модули:")
        for module in arch['medical_modules'][:5]:
            print(f"  • {module}")
    
    if arch['ui_components']:
        print(f"\n🖥️ UI компоненты:")
        for component in arch['ui_components'][:5]:
            print(f"  • {component}")
    
    if arch['data_storage']:
        print(f"\n💾 Хранение данных:")
        for storage in arch['data_storage'][:5]:
            print(f"  • {storage}")
    
    # Предложения по улучшению
    print(f"\n💡 РЕКОМЕНДАЦИИ ПО АРХИТЕКТУРЕ:")
    if summary['python_files'] > 3:
        print("  • Рассмотрите разбиение на модули по функциональности")
    if len(arch['ai_components']) > 1:
        print("  • Создайте общий базовый класс для ИИ-анализаторов")
    if len(arch['medical_modules']) > 2:
        print("  • Выделите медицинскую логику в отдельный пакет")
    if not arch['data_storage']:
        print("  • Добавьте конфигурационные файлы")
    
    # Сохраняем анализ
    output_file = analyzer.save_analysis(tree_data, summary)
    print(f"\n💾 Анализ сохранен в: {output_file}")
    
    print(f"\n✅ Анализ завершен!")

if __name__ == "__main__":
    main()