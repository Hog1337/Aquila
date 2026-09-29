# Короткие команды для типовых действий. `make help` показывает список.
.PHONY: help up down backup restore backup-schedule backup-unschedule backup-status

help:
	@echo "make up            создать .env, собрать образы (нужен интернет) и поднять стек"
	@echo "make down          остановить стек (данные сохраняются)"
	@echo "make backup        холодная резервная копия PostgreSQL, Qdrant и SeaweedFS в backups/"
	@echo "make restore DIR=backups/<копия>   восстановить данные из копии"
	@echo "make backup-schedule   включить ночную копию через cron (BACKUP_DIR, BACKUP_KEEP; см. scripts/backup-schedule.sh)"
	@echo "make backup-unschedule выключить ночную копию"
	@echo "make backup-status     показать расписание и последние копии"

up:
	./run.sh

down:
	./run.sh --down

backup:
	scripts/backup.sh

restore:
	@test -n "$(DIR)" || { echo "укажите DIR=backups/<копия>"; exit 1; }
	scripts/restore.sh "$(DIR)"

backup-schedule:
	scripts/backup-schedule.sh install

backup-unschedule:
	scripts/backup-schedule.sh remove

backup-status:
	scripts/backup-schedule.sh status
