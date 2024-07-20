import React from "react";
import "./NavBar.css";
import { AppBar, Toolbar, Typography, Box, Container } from "@mui/material";

const NavBar = () => {
  return (
    <AppBar position="static" sx={{ boxShadow: 1, width: "100%" }}>
      <Container maxWidth={false} disableGutters>
        <Toolbar sx={{ display: "flex", justifyContent: "center" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <img
              src="https://assets.iu.edu/brand/3.3.x/trident-large.png"
              alt="IU Logo"
              style={{ width: "50px", height: "50px" }}
            />
            <Typography
              variant="h6"
              component="a"
              href="https://bloomington.iu.edu/index.html"
              target="_blank"
              rel="noopener noreferrer"
              sx={{
                fontWeight: "bold",
                textDecoration: "none",
                "&:hover": { textDecoration: "underline" },
              }}
            >
              Indiana University Bloomington
            </Typography>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default NavBar;
