module.exports = {
  apps: [
    {
      name: 'raphael',
      script: 'main.py',
      interpreter: '.venv/bin/python',
      cwd: __dirname,
      restart_delay: 5000,
      max_restarts: 10,
      treekill: true,        // kill child processes (e.g. executor threads) on stop/restart
      env: {
        PYTHONUNBUFFERED: '1',
      },
    },
    {
      name: 'raphael-caffeinate',
      script: 'scripts/start.sh',   // prevent macOS idle sleep — runs independently of bot process
      interpreter: 'bash',
      cwd: __dirname,
      autorestart: false,
      watch: false,
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
