import Logo from "../components/logo";
import SetUpInfo from "../components/setup-info";
import { Stack } from "@mui/material";

function Header() {
  return (
    <Stack
      direction="row"
      width="100%"
      alignItems="center"
      padding="10px"
      sx={{ backgroundColor: "white" }}
      display="flex"
    >
      <Logo />
      <SetUpInfo />
    </Stack>
  );
}

export default Header;
