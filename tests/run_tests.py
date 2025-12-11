#!/usr/bin/env python3
"""
Скрипт для запуска тестов с различными опциями
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], capture_output: bool = False) -> int:
    """
    Запуск команды и возврат кода выхода
    """
    print(f"\n{'='*60}")
    print(f"Запуск команды: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"Ошибки:\n{result.stderr}")
            return result.returncode
        else:
            return subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\nТестирование прервано пользователем")
        return 1
    except Exception as e:
        print(f"Ошибка при запуске команды: {str(e)}")
        return 1


def run_unit_tests(args: argparse.Namespace) -> int:
    """Запуск unit тестов"""
    cmd = [
        "pytest",
        "tests/unit/",
        "-v",
        "--tb=short",
        "-m", "not slow and not integration and not e2e"
    ]
    
    if args.coverage:
        cmd.extend([
            "--cov=api",
            "--cov=src",
            "--cov-report=term",
            "--cov-report=html:coverage_unit"
        ])
    
    return run_command(cmd)


def run_integration_tests(args: argparse.Namespace) -> int:
    """Запуск интеграционных тестов"""
    # Проверяем, запущены ли необходимые сервисы
    if not check_services_running():
        print("Внимание: Некоторые сервисы не запущены.")
        print("Интеграционные тесты могут завершиться с ошибкой.")
        print("Запустите сервисы командой: docker-compose up -d")
        if not args.force:
            response = input("Продолжить? (y/N): ")
            if response.lower() != 'y':
                return 0
    
    cmd = [
        "pytest",
        "tests/integration/",
        "-v",
        "--tb=short",
        "-m", "integration"
    ]
    
    if args.coverage:
        cmd.extend([
            "--cov=api",
            "--cov=src",
            "--cov-append",
            "--cov-report=term",
            "--cov-report=html:coverage_integration"
        ])
    
    return run_command(cmd)


def run_e2e_tests(args: argparse.Namespace) -> int:
    """Запуск end-to-end тестов"""
    print("Проверка запуска всех сервисов...")
    if not check_all_services_running():
        print("Ошибка: Не все сервисы запущены.")
        print("Запустите все сервисы командой: docker-compose up -d")
        return 1
    
    cmd = [
        "pytest",
        "tests/e2e/",
        "-v",
        "--tb=short",
        "-m", "e2e"
    ]
    
    return run_command(cmd)


def run_all_tests(args: argparse.Namespace) -> int:
    """Запуск всех тестов"""
    exit_codes = []
    
    print("Запуск всех тестов...")
    print("\n" + "="*60)
    print("ШАГ 1: Unit тесты")
    print("="*60)
    exit_codes.append(run_unit_tests(args))
    
    print("\n" + "="*60)
    print("ШАГ 2: Интеграционные тесты")
    print("="*60)
    exit_codes.append(run_integration_tests(args))
    
    if args.e2e:
        print("\n" + "="*60)
        print("ШАГ 3: End-to-end тесты")
        print("="*60)
        exit_codes.append(run_e2e_tests(args))
    
    # Сводный отчет
    print("\n" + "="*60)
    print("СВОДНЫЙ ОТЧЕТ")
    print("="*60)
    
    for i, code in enumerate(exit_codes):
        test_type = ["Unit", "Integration", "E2E"][i]
        status = "✓ УСПЕШНО" if code == 0 else "✗ ОШИБКА"
        print(f"{test_type} тесты: {status}")
    
    if any(code != 0 for code in exit_codes):
        print(f"\nОбщий результат: ✗ ТЕСТИРОВАНИЕ НЕ ПРОЙДЕНО")
        return 1
    else:
        print(f"\nОбщий результат: ✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
        return 0


def run_coverage_report(args: argparse.Namespace) -> int:
    """Генерация отчета о покрытии кода"""
    print("Генерация отчета о покрытии кода...")
    
    # Сначала запускаем все тесты с coverage
    cmd_run = [
        "pytest",
        "--cov=api",
        "--cov=src",
        "--cov=feature_store",
        "--cov=streaming",
        "--cov-report=",
        "--cov-append",
        "tests/"
    ]
    
    exit_code = run_command(cmd_run, capture_output=True)
    
    if exit_code != 0 and not args.ignore_failures:
        return exit_code
    
    # Генерируем отчеты
    cmd_report = [
        "coverage",
        "report",
        "--show-missing",
        "--fail-under=60"
    ]
    
    exit_code = run_command(cmd_report)
    
    if exit_code == 0 or args.ignore_failures:
        # Генерируем HTML отчет
        cmd_html = [
            "coverage",
            "html",
            "--directory=coverage_report",
            "--title='Predictive Maintenance Coverage Report'"
        ]
        run_command(cmd_html)
        
        # Генерируем XML отчет (для CI/CD)
        cmd_xml = ["coverage", "xml", "-o", "coverage.xml"]
        run_command(cmd_xml)
        
        print("\nОтчеты о покрытии:")
        print(f"- HTML: file://{os.path.abspath('coverage_report/index.html')}")
        print(f"- XML: {os.path.abspath('coverage.xml')}")
    
    return exit_code if not args.ignore_failures else 0


def run_performance_tests(args: argparse.Namespace) -> int:
    """Запуск тестов производительности"""
    print("Запуск тестов производительности...")
    
    # Тесты нагрузки
    cmd_load = [
        "pytest",
        "tests/performance/test_load.py",
        "-v",
        "--tb=no"
    ]
    
    exit_code = run_command(cmd_load)
    
    if exit_code == 0:
        print("\n" + "="*60)
        print("ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ ЗАВЕРШЕНЫ")
        print("="*60)
    
    return exit_code


def run_specific_test(args: argparse.Namespace) -> int:
    """Запуск конкретного теста"""
    test_path = args.test_path
    
    if not os.path.exists(test_path):
        # Пробуем найти в tests директории
        test_path = os.path.join("tests", test_path)
        if not os.path.exists(test_path):
            print(f"Тест не найден: {args.test_path}")
            return 1
    
    cmd = ["pytest", test_path, "-v", "--tb=short"]
    
    if args.coverage:
        cmd.extend(["--cov=api", "--cov-report=term"])
    
    return run_command(cmd)


def check_services_running() -> bool:
    """Проверка запущены ли необходимые сервисы"""
    try:
        # Проверяем PostgreSQL
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="password",
            connect_timeout=2
        )
        conn.close()
        
        # Проверяем Redis
        import redis
        r = redis.Redis(host="localhost", port=6379, socket_timeout=2)
        r.ping()
        
        return True
    except:
        return False


def check_all_services_running() -> bool:
    """Проверка запуска всех сервисов"""
    services = check_services_running()
    
    # Дополнительные проверки
    try:
        # Проверяем Kafka
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            max_block_ms=2000
        )
        producer.close()
        return services
    except:
        return False


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Скрипт для запуска тестов Predictive Maintenance системы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s all                    # Запуск всех тестов
  %(prog)s unit                   # Только unit тесты
  %(prog)s integration           # Только интеграционные тесты
  %(prog)s coverage              # Генерация отчета о покрытии
  %(prog)s run tests/test_api.py # Запуск конкретного теста
  %(prog)s all --coverage        # Все тесты с покрытием
  %(prog)s all --e2e            # Все тесты включая E2E
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Команда")
    
    # Парсер для всех тестов
    parser_all = subparsers.add_parser("all", help="Запуск всех тестов")
    parser_all.add_argument("--coverage", action="store_true", help="Включить coverage")
    parser_all.add_argument("--e2e", action="store_true", help="Включить E2E тесты")
    parser_all.add_argument("--force", action="store_true", help="Пропустить проверку сервисов")
    
    # Парсер для unit тестов
    parser_unit = subparsers.add_parser("unit", help="Запуск unit тестов")
    parser_unit.add_argument("--coverage", action="store_true", help="Включить coverage")
    
    # Парсер для интеграционных тестов
    parser_integration = subparsers.add_parser("integration", help="Запуск интеграционных тестов")
    parser_integration.add_argument("--coverage", action="store_true", help="Включить coverage")
    parser_integration.add_argument("--force", action="store_true", help="Пропустить проверку сервисов")
    
    # Парсер для E2E тестов
    parser_e2e = subparsers.add_parser("e2e", help="Запуск end-to-end тестов")
    
    # Парсер для coverage
    parser_coverage = subparsers.add_parser("coverage", help="Генерация отчета о покрытии")
    parser_coverage.add_argument("--ignore-failures", action="store_true", 
                                help="Игнорировать проваленные тесты")
    
    # Парсер для производительности
    parser_perf = subparsers.add_parser("performance", help="Запуск тестов производительности")
    
    # Парсер для запуска конкретного теста
    parser_run = subparsers.add_parser("run", help="Запуск конкретного теста")
    parser_run.add_argument("test_path", help="Путь к тесту")
    parser_run.add_argument("--coverage", action="store_true", help="Включить coverage")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Выполнение команды
    commands = {
        "all": run_all_tests,
        "unit": run_unit_tests,
        "integration": run_integration_tests,
        "e2e": run_e2e_tests,
        "coverage": run_coverage_report,
        "performance": run_performance_tests,
        "run": run_specific_test
    }
    
    exit_code = commands[args.command](args)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()