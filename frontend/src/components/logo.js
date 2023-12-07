import React from "react";
import { Stack, Typography } from "@mui/material";
import "../index.css";

function Logo() {
  return (
    <Stack direction="row" spacing={2} alignItems="center" width="100%">
      <img
        src="https://i.pinimg.com/originals/05/91/c7/0591c7d9ed972c451f02e9d52199f1d6.gif"
        alt="Logo"
        className="logo"
      />
      <Typography variant="h5" className="title">
        generative disco
      </Typography>
    </Stack>
  );
}

export default Logo;
