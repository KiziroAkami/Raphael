module.exports = {
  apps: [
    {
      name: 'raphael',
      script: 'scripts/start.sh',   // caffeinate wrapper — prevents macOS idle sleep
      interpreter: 'bash',
      cwd: __dirname,
      restart_delay: 5000,
      max_restarts: 10,
      env: {
        PYTHONUNBUFFERED: '1',
      },
    },
    {
      name: 'raphael-sync',
      script: 'scripts/build_index.py',
      interpreter: '.venv/bin/python',
      cwd: __dirname,
      cron_restart: '0 3 * * *',    // run incremental wiki sync daily at 03:00
      autorestart: false,            // one-shot job — do not restart on exit
      watch: false,
      args: '--incremental',
      env: {
        PYTHONUNBUFFERED: '1',
      },
    },
  ],
};
